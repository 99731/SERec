#include "eval.h"
#include "utils.h"
#include <omp.h>
using namespace std;

eval::eval(){
    num_factors = 0; // m_num_topics
    num_items = 0; // m_num_docs
    num_users = 0; // num of users
    best_mae = 9999;
    best_rsme = 9999;

}

eval::~eval() {
  // free memory
 
}
void eval::set_parameters(char* directory,int num_factors,int num_users,int num_items){
	this->num_factors = num_factors;
    this->num_users = num_users;
    this->num_items = num_items;

	// initializing the saving file
	strcpy(mae_path,directory);
	strcat(mae_path,"/mae.txt");
	strcpy(rsme_path,directory);
	strcat(rsme_path,"/rsme.txt");
}


// calculate the mae and rsme
double eval::cal_mae( ){
	FILE *resultFile;

	resultFile = fopen(mae_path,"w");
	fprintf(resultFile,"\nThe mae is %lf.\n",best_mae);
	fclose(resultFile);
	return best_mae;
}
double eval::cal_rsme(){
	FILE *resultFile;

	resultFile = fopen(rsme_path,"w");
	fprintf(resultFile,"\nThe rsme is %lf.\n",best_rsme);
	fclose(resultFile);
	return best_rsme;
}


double eval::mtx_mae(gsl_matrix* m_U,gsl_matrix* m_V, const r_data* test_users,int num_test_users)
{
	double mae = 0.0,sum = 0.0,tRate = 0.0,iRate = 0.0;
	int i, j,l,n,s,cnt = 0;
	int* item_ids; 
  	double* item_scores;
	
	for (i = 0; i < num_test_users; i ++) {
		gsl_vector_view u = gsl_matrix_row(m_U, i);
		item_ids = test_users->m_vec_data[i];
	     	item_scores = test_users->m_vec_score[i];

    		n = test_users->m_vec_len[i];
		if (n > 0) { // this user has rated some articles
			for (l=0; l < n; l ++) {
			  j = item_ids[l];
			  s = item_scores[l];
			  tRate = s/1.0;
			  gsl_vector_view v = gsl_matrix_row(m_V, j);
			  gsl_blas_ddot(&u.vector, &v.vector, &iRate);
			  sum +=  fabs(tRate - iRate);cnt++;
			}
		}

	}

	mae = sum/cnt;
	if (mae<best_mae) best_mae=mae;
	printf("the mae is: %lf \n",best_mae);
	return mae;
}

double eval::mtx_rsme(gsl_matrix* m_U,gsl_matrix* m_V, const r_data* test_users,int num_test_users)
{
	double rsme = 0.0,sum = 0.0,tRate = 0.0,iRate = 0.0;
	int i, j,l,n,s,cnt = 0;
	int* item_ids; 
  	double* item_scores;
	for (i = 0; i < num_test_users; i ++) {
		gsl_vector_view u = gsl_matrix_row(m_U, i);
		item_ids = test_users->m_vec_data[i];
     	item_scores = test_users->m_vec_score[i];

    	n = test_users->m_vec_len[i];
		if (n > 0) { // this user has rated some articles
			for (l=0; l < n; l ++) {
			  j = item_ids[l];
			  s = item_scores[l];
			  tRate = s/1.0;
			  gsl_vector_view v = gsl_matrix_row(m_V, j);
			  gsl_blas_ddot(&u.vector, &v.vector, &iRate);
			  sum = sum + (tRate - iRate)*(tRate - iRate);
			  cnt++;
			}
		}

	}

	rsme = sqrt(sum/cnt);if (rsme<best_rsme) best_rsme=rsme;
	return rsme;
}

void cal_log4ndcg(double *result, int k){
	//double result[k]; 
	for(int i = 1; i<=k; i++){
		result[i-1] = 1.0/log2(i+1);//fast_log2(i+1);
	}
	//return result;
}

double eval::recall_at_k(const gsl_matrix* m_U,const gsl_matrix* m_V, const r_data* train_user,
	const r_data* vali_user, const r_data* test_user,int num_users, int k){
	int state, count, hit,zero_count = 0;
	double recall = 0;
	size_t index;
	int n_threads = 20;
	FILE * file_r = fopen("predict.dat","w");
//	gsl_vector* x  = gsl_vector_alloc(m_V->size1);
//	gsl_vector* u  = gsl_vector_alloc(m_V->size2);
//	gsl_vector* item_lookup  = gsl_vector_alloc(m_V->size1);
//	gsl_permutation * sorted_indices = gsl_permutation_alloc (m_V->size1);
	#pragma omp parallel for schedule(dynamic, num_users/n_threads +1)reduction(+:recall)
	for(int i = 0; i < num_users; i++){
//		printf("recall iteration no. %d\n",i);
		int min_d = 0;
		gsl_vector* x  = gsl_vector_alloc(m_V->size1);
		gsl_vector* u  = gsl_vector_alloc(m_V->size2);
		gsl_vector* item_lookup  = gsl_vector_alloc(m_V->size1);
		gsl_permutation * sorted_indices = gsl_permutation_alloc (m_V->size1);
		
		gsl_vector_set_zero(x);
		gsl_vector_set_zero(item_lookup);
		
		if(0 == test_user->m_vec_len[i]){
			zero_count += 1;
			continue;
		}
		// use item_lookup to record wether the item is rated 
		//0 if not rated, 1 in train/vali data, 2 in test data
		int* item_ids = train_user->m_vec_data[i];
		for(int j = 0; j < train_user->m_vec_len[i]; j++){
			gsl_vector_set(item_lookup, item_ids[j], 1);

		}
		item_ids = vali_user->m_vec_data[i];
		for(int j = 0; j < vali_user->m_vec_len[i]; j++){
			gsl_vector_set(item_lookup, item_ids[j], 1);

		}
		item_ids = test_user->m_vec_data[i];
		for(int j = 0; j < test_user->m_vec_len[i]; j++){
			gsl_vector_set(item_lookup, item_ids[j], 2);

		}
		
		//printf("recall step1..%ld,%ld,%ld\n",m_V->size1, m_V->size2,u->size);	
		gsl_matrix_get_row(u, m_U, i); 
	
		gsl_blas_dgemv(CblasNoTrans, 1.0, m_V, u, 0.0, x);
		
		// print to ratings to file
		if(i<50){
			vct_fprintf(file_r,x);
		}
		gsl_sort_vector_index(sorted_indices, x); // second para should be const gsl_vector
		gsl_permutation_reverse(sorted_indices);
		count = 0;
		hit = 0;
		bool flag = true;
		for(size_t order = 0; order < gsl_permutation_size(sorted_indices) && flag; order++){
			index = gsl_permutation_get(sorted_indices, order);
			state = gsl_vector_get(item_lookup,index);
			
			if(state == 0){
				count ++;

			}else if(state == 1){
				continue;
			}else if(state == 2){
				// make a hit
				count ++;
				hit ++;
			}else{
				printf("there is an unexpected value in item_lookup...!\n");
				exit(-1);
			}

			if(count == k){
				min_d = test_user->m_vec_len[i];
				if(min_d > k){
					min_d = k;
				}
				recall += 1.0 * hit / min_d;
				//break;
				flag = false;
			}
		}
	gsl_permutation_free(sorted_indices);
	gsl_vector_free(x);
	gsl_vector_free(u);
	gsl_vector_free(item_lookup);
	}
//	gsl_permutation_free(sorted_indices);
//	gsl_vector_free(x);
//	gsl_vector_free(u);
//	gsl_vector_free(item_lookup);
	fclose(file_r);
	best_recall = recall / (num_users - zero_count);
	printf("the recall@%d is %lf\n",k, best_recall);
	return best_recall;
}

double eval::recall_at_k_withmu(const gsl_matrix* m_MU, const gsl_matrix* m_U,const gsl_matrix* m_V, const r_data* train_user,
	const r_data* vali_user, const r_data* test_user,int num_users, int k){
	int state, count, hit,zero_count = 0;
	double recall = 0;
	size_t index;
	int n_threads = 20;
	FILE * file_r = fopen("predict.dat","w");
//	gsl_vector* x  = gsl_vector_alloc(m_V->size1);
//	gsl_vector* u  = gsl_vector_alloc(m_V->size2);
//	gsl_vector* item_lookup  = gsl_vector_alloc(m_V->size1);
//	gsl_permutation * sorted_indices = gsl_permutation_alloc (m_V->size1);
	#pragma omp parallel for schedule(dynamic, num_users/n_threads +1)reduction(+:recall)
	for(int i = 0; i < num_users; i++){
//		printf("recall iteration no. %d\n",i);
		int min_d = 0;
		gsl_vector* row_mu  = gsl_vector_alloc(m_V->size1);
		gsl_vector* x  = gsl_vector_alloc(m_V->size1);
		gsl_vector* u  = gsl_vector_alloc(m_V->size2);
		gsl_vector* item_lookup  = gsl_vector_alloc(m_V->size1);
		gsl_permutation * sorted_indices = gsl_permutation_alloc (m_V->size1);
		
		gsl_vector_set_zero(x);
		gsl_vector_set_zero(item_lookup);
		
		if(0 == test_user->m_vec_len[i]){
			zero_count += 1;
			continue;
		}
		// use item_lookup to record wether the item is rated 
		//0 if not rated, 1 in train/vali data, 2 in test data
		int* item_ids = train_user->m_vec_data[i];
		for(int j = 0; j < train_user->m_vec_len[i]; j++){
			gsl_vector_set(item_lookup, item_ids[j], 1);//Revised ,should be 1

		}
		item_ids = vali_user->m_vec_data[i];
		for(int j = 0; j < vali_user->m_vec_len[i]; j++){
			gsl_vector_set(item_lookup, item_ids[j], 1);//Revised ,should be 1

		}
		item_ids = test_user->m_vec_data[i];
		for(int j = 0; j < test_user->m_vec_len[i]; j++){
			gsl_vector_set(item_lookup, item_ids[j], 2);

		}
		
		//printf("recall step1..%ld,%ld,%ld\n",m_V->size1, m_V->size2,u->size);	
		gsl_matrix_get_row(u, m_U, i); //with mu
		gsl_matrix_get_row(row_mu, m_MU, i);
		
		gsl_blas_dgemv(CblasNoTrans, 1.0, m_V, u, 0.0, x);
		gsl_vector_mul(x,row_mu); //with mu
		// print to ratings to file
		if(i<50){
			vct_fprintf(file_r,x);
		}
		gsl_sort_vector_index(sorted_indices, x); // second para should be const gsl_vector
		gsl_permutation_reverse(sorted_indices);
		count = 0;
		hit = 0;
		bool flag = true;
		for(size_t order = 0; order < gsl_permutation_size(sorted_indices) && flag; order++){
			index = gsl_permutation_get(sorted_indices, order);
			state = gsl_vector_get(item_lookup,index);
			
			if(state == 0){
				count ++;

			}else if(state == 1){
				continue;
			}else if(state == 2){
				// make a hit
				count ++;
				hit ++;
			}else{
				printf("there is an unexpected value in item_lookup...!\n");
				exit(-1);
			}

			if(count == k){
				min_d = test_user->m_vec_len[i];
				if(min_d > k){
					min_d = k;
				}
				recall += 1.0 * hit / min_d;
				//break;
				flag = false;
			}
		}
	gsl_permutation_free(sorted_indices);
	gsl_vector_free(x);
	gsl_vector_free(u);
	gsl_vector_free(item_lookup);
	}
//	gsl_permutation_free(sorted_indices);
//	gsl_vector_free(x);
//	gsl_vector_free(u);
//	gsl_vector_free(item_lookup);
	fclose(file_r);
	best_recall = recall / (num_users - zero_count);
	printf("the recall@%d is %lf, and the zero count is %d\n",k, best_recall, zero_count);
	return best_recall;
}


double eval::map_at_k(const gsl_matrix* m_U,const gsl_matrix* m_V, const r_data* train_user,
	const r_data* vali_user, const r_data* test_user,int num_users, int k){
	int state, zero_count = 0;
	double map = 0;
	size_t index;
	int n_threads = 20;
//	FILE * file_r = fopen("predict.dat","w");
//	gsl_vector* x  = gsl_vector_alloc(m_V->size1);
//	gsl_vector* u  = gsl_vector_alloc(m_V->size2);
//	gsl_vector* item_lookup  = gsl_vector_alloc(m_V->size1);
//	gsl_permutation * sorted_indices = gsl_permutation_alloc (m_V->size1);
	#pragma omp parallel for schedule(dynamic, num_users/n_threads +1)reduction(+:map)
	for(int i = 0; i < num_users; i++){
		
		double ap = 0;
		int count = 0, hit = 0;//, min_d;
		gsl_vector* x  = gsl_vector_alloc(m_V->size1);
		gsl_vector* u  = gsl_vector_alloc(m_V->size2);
		gsl_vector* item_lookup  = gsl_vector_alloc(m_V->size1);
		gsl_permutation * sorted_indices = gsl_permutation_alloc (m_V->size1);
		
		gsl_vector_set_zero(x);
		gsl_vector_set_zero(item_lookup);
		
		if(0 == test_user->m_vec_len[i]){
			zero_count += 1;
			continue;
		}
		// use item_lookup to record wether the item is rated 
		//0 if not rated, 1 in train/vali data, 2 in test data
		int* item_ids = train_user->m_vec_data[i];
		for(int j = 0; j < train_user->m_vec_len[i]; j++){
			gsl_vector_set(item_lookup, item_ids[j], 1);

		}
		item_ids = vali_user->m_vec_data[i];
		for(int j = 0; j < vali_user->m_vec_len[i]; j++){
			gsl_vector_set(item_lookup, item_ids[j], 1);

		}
		item_ids = test_user->m_vec_data[i];
		for(int j = 0; j < test_user->m_vec_len[i]; j++){
			gsl_vector_set(item_lookup, item_ids[j], 2);

		}
		
		//printf("recall step1..%ld,%ld,%ld\n",m_V->size1, m_V->size2,u->size);	
		gsl_matrix_get_row(u, m_U, i); 
	
		gsl_blas_dgemv(CblasNoTrans, 1.0, m_V, u, 0.0, x);
		
		// print to ratings to file
		//if(i<50){
		//	vct_fprintf(file_r,x);
		//}
		gsl_sort_vector_index(sorted_indices, x); // second para should be const gsl_vector
		gsl_permutation_reverse(sorted_indices);
		count = 0;
		hit = 0;
		bool flag = true;
		for(size_t order = 0; order < gsl_permutation_size(sorted_indices) && flag; order++){
			index = gsl_permutation_get(sorted_indices, order);
			state = gsl_vector_get(item_lookup,index);
			
			if(state == 0){
				count ++;
				ap += 1.0 * hit / count;

			}else if(state == 1){
				continue;
			}else if(state == 2){
				// make a hit
				count ++;
				hit ++;
				ap += 1.0 * hit / count;
			}else{
				printf("there is an unexpected value in item_lookup...!\n");
				exit(-1);
			}

			if(count == k){
				
				//min_d = test_user->m_vec_len[i];
				//if(min_d > k){
				//	min_d = k;
				//}
				//map += ap / min_d;
				map += ap / k;
				flag = false;
			}
		}
	gsl_permutation_free(sorted_indices);
	gsl_vector_free(x);
	gsl_vector_free(u);
	gsl_vector_free(item_lookup);
	}
//	gsl_permutation_free(sorted_indices);
//	gsl_vector_free(x);
//	gsl_vector_free(u);
//	gsl_vector_free(item_lookup);
	
	best_map = map / (num_users - zero_count);
	printf("the map@%d is %lf\n",k, best_map);
	return best_map;
}

double eval::ndcg_at_k(const gsl_matrix* m_U,const gsl_matrix* m_V, const r_data* train_user,
	const r_data* vali_user, const r_data* test_user,int num_users, int k){
	double result[k];
	double ndcg = 0;
	int zero_count = 0;
	size_t index;

	int n_threads = 20;
	cal_log4ndcg(result,k);
	#pragma omp parallel for schedule(dynamic, num_users/n_threads +1)reduction(+:ndcg)
	for(int i = 0; i < num_users; i++){
		double dcg = 0, idcg = 0;
		int count, state ;
		gsl_vector* x  = gsl_vector_alloc(m_V->size1);
		gsl_vector* u  = gsl_vector_alloc(m_V->size2);
		gsl_vector* item_lookup  = gsl_vector_alloc(m_V->size1);
		gsl_permutation * sorted_indices = gsl_permutation_alloc (m_V->size1);
	
		gsl_vector_set_zero(x);
		gsl_vector_set_zero(item_lookup);
	
		
		if(0 == test_user->m_vec_len[i]){
			zero_count += 1;
			continue;
		}
		// use item_lookup to record wether the item is rated 
		//0 if not rated, 1 in train/vali data, 2 in test data
		int* item_ids = train_user->m_vec_data[i];
		for(int j = 0; j < train_user->m_vec_len[i]; j++){
			gsl_vector_set(item_lookup, item_ids[j], 1);

		}
		item_ids = vali_user->m_vec_data[i];
		for(int j = 0; j < vali_user->m_vec_len[i]; j++){
			gsl_vector_set(item_lookup, item_ids[j], 1);

		}
		item_ids = test_user->m_vec_data[i];
		for(int j = 0; j < test_user->m_vec_len[i]; j++){
			gsl_vector_set(item_lookup, item_ids[j], 2);

		}

		gsl_matrix_get_row(u, m_U, i); 
		gsl_blas_dgemv(CblasNoTrans, 1.0, m_V, u, 0.0, x);
		gsl_sort_vector_index(sorted_indices, x); // second para should be const gsl_vector
		gsl_permutation_reverse(sorted_indices);
		count = 0;
		//hit = 0;
		bool flag = true;
		for(size_t order = 0; order < gsl_permutation_size(sorted_indices) && flag; order++){
			
			index = gsl_permutation_get(sorted_indices, order);
			state = gsl_vector_get(item_lookup,index);
		
			if(state == 0){
				count ++;

			}else if(state == 1){
				continue;
			}else if(state == 2){
				// make a hit
				dcg += result[count];
				count ++;
				//hit ++;

			}else{
				printf("there is an unexpected value in item_lookup...!\n");
				exit(-1);
			}

			if(count == k){
				//recall += 1.0 * hit / test_user->m_vec_len[i];
				//break;
				flag = false;
			}
		}

		for(int j = 0; j < test_user->m_vec_len[i]; j++){
			idcg += result[j];
			//idcg += 1/result[j];
		}
		ndcg += dcg / idcg;

		gsl_permutation_free(sorted_indices);
		gsl_vector_free(x);
		gsl_vector_free(u);
		gsl_vector_free(item_lookup);
	}
	//gsl_permutation_free(sorted_indices);
	//gsl_vector_free(x);
	//gsl_vector_free(u);
	//gsl_vector_free(item_lookup);

	
	best_ndcg= ndcg/(num_users - zero_count);
	printf("the ndcg@%d is %lf\n",k, best_ndcg);
	return best_ndcg;
}

