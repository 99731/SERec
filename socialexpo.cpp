#include "socialexpo.h"
#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;
//int n_threads = 20;

s_expo::s_expo(){

	m_beta = NULL;
	m_theta = NULL;
	m_mu = NULL;
	n_users = 0;
	n_items = 0;

	X = NULL;
	T = NULL;
	B = NULL;
}

s_expo::~s_expo(){
	if (m_beta != NULL) gsl_matrix_free(m_beta);
	if (m_theta != NULL) gsl_matrix_free(m_theta);
	if (m_mu != NULL) gsl_matrix_free(m_mu);
	if (X != NULL) gsl_matrix_free(X);
	if (T != NULL) gsl_matrix_free(T);
	if (B != NULL) gsl_matrix_free(B);
	
}

void s_expo::init_para(const int users, const int items, const sexpo_hyperparameter* shyper){

	m_theta = gsl_matrix_alloc(users,shyper->n_components);
	double init_std = shyper->init_std;
	double init_mu = shyper->init_mu;
	n_users = users;
	n_items = items;
	printf("m_theta size %ld,%ld\n",m_theta->size1,m_theta->size2);
	for (size_t i = 0; i < m_theta->size1; i++)
    	for (size_t j = 0; j < m_theta->size2; j++){
      		mset(m_theta, i, j, init_std *runiform()); 
	}
    printf("m_theta size %ld,%ld\n", m_theta->size1,m_theta->size2);

    m_beta = gsl_matrix_alloc(items, shyper->n_components);
	for (size_t i = 0; i < m_beta->size1; i++)
    	for (size_t j = 0; j < m_beta->size2; j++)
      		mset(m_beta, i, j, init_std *runiform());
    printf("m_beta size %ld,%ld\n", m_beta->size1,m_beta->size2);

    m_mu = gsl_matrix_alloc(users, items);
	for (size_t i = 0; i < m_mu->size1; i++)
    	for (size_t j = 0; j < m_mu->size2; j++)
      		mset(m_mu, i, j, init_mu);
    printf("m_mu size %ld,%ld\n", m_mu->size1,m_mu->size2);

}

void s_expo::init_para_sr(const sexpo_hyperparameter* shyper){
	
	int sr_k = shyper->sr_k;
	X = gsl_matrix_alloc(n_users, sr_k);
	T = gsl_matrix_alloc(n_items, sr_k);
	B = gsl_matrix_alloc(n_users, sr_k);
	printf("initializing para sr...\n");
	for (size_t i = 0; i < X->size1; i++)
    	for (size_t j = 0; j < X->size2; j++)
      		mset(X, i, j, 0.01*runiform());

	for (size_t i = 0; i < T->size1; i++)
    	for (size_t j = 0; j < T->size2; j++)
      		mset(T, i, j, 0.01*runiform());

	for (size_t i = 0; i < B->size1; i++)
    	for (size_t j = 0; j < B->size2; j++)
      		mset(B, i, j, 0.01*runiform());
	printf("para sr initialized..\n");
}
void s_expo::fit(const r_data* users, const r_data* items, const r_data* social, 
	const r_data* vad_data, const sexpo_hyperparameter* shyper){
	//init_para(int users, int items, shyper) //TO DO
	//update()
	omp_set_num_threads(20);
//	printf("it is in the fitting stage...\n");

	time_t start, current;
	time(&start);
	int elapsed = 0;
	int iter = 0;
	double converge = 1.0;
	//double last_ndcg = -exp(50);
  	double likelihood = -exp(50),likelihood_old;
	/*
	if (shyper->version == 3){
		
		double mu_d = fast_log(shyper->init_mu/(1 - shyper->init_mu));
		for (size_t i = 0; i < m_mu->size1; i++)
    		for (size_t j = 0; j < m_mu->size2; j++)
      			mset(m_mu, i, j, mu_d);
	}
	*/
	while (iter < shyper->max_iter and converge > 1e-6 ){
		if(shyper->verbose){
			printf("iteration #%d\n",iter);
		}
		likelihood_old = likelihood;
		likelihood  = update_factors(users,items,social,shyper, iter);
		converge = fabs( (likelihood - likelihood_old) / likelihood_old);
		iter += 1;
		time(&current);
		elapsed = (int)difftime(current,start);
		printf("iter = %d, time = %06d, likelihood = %.5f, converge = %.10f\n",
			iter, elapsed, likelihood, converge);
	}



}



double s_expo::update_factors(const r_data* users, const r_data* items,const r_data* social_relation,  const sexpo_hyperparameter* shyper,const int iter){
	//time_t start, current;
  	//time(&start);
//	if(shyper->verbose){
//			printf("updating  factors ...\n");
//	}
	double likelihood = 0;
	//size_t i,j;
	int n_threads = 20;
	gsl_matrix* P  = gsl_matrix_alloc(n_users, n_items);
	//double result,pex,mu;
	//updating mu
	//printf("calculating P...\n");
	if(shyper->verbose){
		printf("calculating P....\n");
	}
	#pragma omp parallel for schedule(dynamic, n_users/n_threads +1)  reduction(+:likelihood)
	for (size_t i = 0; i < m_mu->size1; i++){
		double result;
		int flags[m_mu->size2]  = {};
		int n = users->m_vec_len[i];
		//int* item_ids = users->m_vec_data[i];
		int* item_ids = users->m_vec_data[i];
		double* item_scores = users->m_vec_score[i];
		for (int l=0; l < n; l ++) {
			if(item_scores[l] == 0){
				flags[item_ids[l]] = 2;	
			}else if(item_scores[l] == 1){
				flags[item_ids[l]] = 1;
			}
		}  
		
		 
		for (size_t j = 0; j < m_mu->size2; j++){
			gsl_vector_const_view u = gsl_matrix_const_row(m_theta, i);
			gsl_vector_const_view v = gsl_matrix_const_row(m_beta, j);
			// notice m_mu = gsl_matrix_alloc(n_items, n_users);
			gsl_blas_ddot(&u.vector, &v.vector, &result);
			double t_mu = gsl_matrix_get(m_mu, i,j);
			if(flags[j] == 2){
				// for the case that y = 0
				result = normal_pdf(0, result, shyper->sigma_y, t_mu);
			}else{
				result = normal_pdf(flags[j], result, shyper->sigma_y, t_mu);
			}
			
			if(iter > -1){
				if(result < 1&&result>0){
					}
				else
				result = 0;
				//erintf("%.5f\n",result);
			}

			if(flags[j] == 0){
				likelihood += fast_log(t_mu * result + 1 - t_mu);
				result = result / (result + (1 - t_mu) / t_mu);
			}else{			
				likelihood += fast_log(t_mu * result);
				result = 1;
			}
			gsl_matrix_set(P, i, j, result);
		}

	}
//	printf("likelihood 1 is %.5f\n",likelihood);
	//updating user theta
	
	//printf("updating user factors m_theta...\n");
	if(shyper->verbose){
		printf("updating user factors m_theta....\n");
	}
	double theta_y = shyper->lam_theta / shyper->lam_y;
	
	#pragma omp parallel for schedule(dynamic, n_users/n_threads +1) reduction(+:likelihood)
	//#pragma omp parallel for reduction(+:likelihood)
	for ( size_t i = 0; i < m_mu->size1; i++){
		gsl_matrix* A  = gsl_matrix_alloc(m_theta->size2, m_theta->size2);
		gsl_vector* x  = gsl_vector_alloc(m_theta->size2);
		gsl_matrix_set_zero(A);
		for (size_t  j = 0; j < m_mu->size2; j++){
			
			gsl_vector_const_view v = gsl_matrix_const_row(m_beta, j);
			gsl_blas_dger(gsl_matrix_get(P, i, j), &v.vector, &v.vector, A);
		}
		int n = users->m_vec_len[i];
		int rated_id = -1;
		gsl_vector_set_zero(x);
		//if (n > 0) {
		 // this user has rated some articles
        	//double* item_scores = users->m_vec_score[i];
        	int* item_ids = users->m_vec_data[i];

//        	printf("updating user factors theta step1...\n");

        	for (int l=0; l < n; l ++) {
         		rated_id = item_ids[l];
			if(rated_id >=n_items){
				printf("id %d is out of index, n_items:%d\n",rated_id, n_items);
			}
          		gsl_vector_const_view v = gsl_matrix_const_row(m_beta, rated_id);
          		gsl_blas_daxpy(gsl_matrix_get(P, i, rated_id), &v.vector, x); 
        	}


		gsl_matrix_add_diagonal(A, theta_y);
		gsl_vector_view u = gsl_matrix_row(m_theta, i);
        

        	// update likelihood add theta * theta
        	double likelihood_fragment;
        	gsl_blas_ddot(&u.vector, &u.vector, &likelihood_fragment);
        	likelihood += -0.5 * shyper->lam_theta * likelihood_fragment;

        	matrix_vector_solve(A, x, &(u.vector));
        	gsl_matrix_free(A);
    		gsl_vector_free(x);
	}
	
//	printf("likelihood 2 is %.5f\n",likelihood);

	// updating item m_beta
	//printf("updating item factors m_beta...\n");
	if(shyper->verbose){
		printf("updating item factors m_beta....\n");
	}
	double beta_y = shyper->lam_beta / shyper->lam_y;
	
	#pragma omp parallel for schedule(dynamic, n_users/n_threads +1)  reduction(+:likelihood)
	for ( size_t i = 0; i < m_mu->size2; i++){
		
	//	if(shyper->verbose){
	//		printf("m_beta loop iter..%ld\n",i);
	//	}
		gsl_matrix* A  = gsl_matrix_alloc(m_theta->size2, m_theta->size2);
	  	gsl_vector* x  = gsl_vector_alloc(m_theta->size2);
		gsl_matrix_set_zero(A);
	//	printf("updating m_beta setp1..\n");        
		for ( size_t j = 0; j < m_theta->size1; j++){
			
			gsl_vector_const_view u = gsl_matrix_const_row(m_theta, j);
			gsl_blas_dger(gsl_matrix_get(P, j, i), &u.vector, &u.vector, A);
		}
		//int n = users->m_vec_len[i];
		int m = items->m_vec_len[i];
		int rated_id = -1;
		gsl_vector_set_zero(x);
	        int* user_ids = items->m_vec_data[i];
		
	//	printf("updating m_beta setp2..\n");        
        	for (int l=0; l < m; l ++) {
          		rated_id = user_ids[l];
		//	printf("l is %d and rated_id is %d..\n",l,rated_id);        
          		gsl_vector_const_view u = gsl_matrix_const_row(m_theta, rated_id);
          		gsl_blas_daxpy(gsl_matrix_get(P, rated_id, i), &u.vector, x); 
        	}


		gsl_matrix_add_diagonal(A, beta_y);
		gsl_vector_view v = gsl_matrix_row(m_beta, i);
        

	//	printf("updating m_beta setp3..\n");        
        	// update likelihood add beta * beta before update beta
        	double likelihood_fragment;
        	gsl_blas_ddot(&v.vector, &v.vector, &likelihood_fragment);
        	likelihood += -0.5 * shyper->lam_beta * likelihood_fragment;

        	matrix_vector_solve(A, x, &(v.vector));
        	gsl_matrix_free(A);
    		gsl_vector_free(x);
	}
	
//	printf("likelihood 3  is %.5f\n",likelihood);
	// update mu
	//printf("updating mu...\n");
	if(shyper->verbose){
		printf("updating mu...\n");
	}
	
	if(shyper->version == 1){
		// --version 1 - purely based on item popularity
		#pragma omp parallel for schedule(dynamic, n_users/n_threads +1) 
		for(size_t i = 0; i < m_mu->size2; i++){
			double new_mu;
			//gsl_matrix_const_view v = gsl_matrix_const_column(P,i);
			gsl_vector* v = gsl_vector_alloc(m_mu->size1);
			gsl_matrix_get_col(v,P,i);
			new_mu = gsl_blas_dasum(v);
			new_mu = (shyper->a - 1 + new_mu) / (shyper->a + shyper->b + m_mu->size1 - 2);

			for(size_t uid = 0; uid < m_mu->size1; uid++){
				gsl_matrix_set(m_mu, uid, i, new_mu);
			}
			gsl_vector_free(v);

		}
		// -- version 1 end

	}else if(shyper->version == 2){
		//-- version 2 begin, using social boosting
		#pragma omp parallel for schedule(dynamic, n_users/n_threads +1) 
		for(size_t i = 0; i < m_mu->size2; i++){
			double new_mu,  denominator;
			//int friend_id, friend_num;
			//int* friend_ids;
			//gsl_matrix_const_view v = gsl_matrix_const_column(P,i);
			gsl_vector* v = gsl_vector_alloc(m_mu->size1);
			gsl_matrix_get_col(v,P,i);
			new_mu = gsl_blas_dasum(v);
		
			denominator = shyper->a + shyper->b + m_mu->size1 - 2;
			
			for(size_t uid = 0; uid < m_mu->size1; uid++){
				int friend_id, friend_num;
				int* friend_ids;
				double social_accum = 0.0,result;
				friend_num = social_relation->m_vec_len[uid];
				if(friend_num > 0){
					friend_ids = social_relation->m_vec_data[uid];
					for(int  k = 0; k < friend_num; k++){
						friend_id = friend_ids[k];
						social_accum += gsl_matrix_get(P,uid,friend_id);
					}
					social_accum *= (shyper->s -1);
				}
		
			//	printf("popularity: %lf social_accum: %lf\n",new_mu,social_accum);
			//	if(uid>50)break;
				result = (shyper->a - 1 + new_mu + social_accum)/(social_accum + denominator);
				gsl_matrix_set(m_mu, uid, i, result);
			}
		
			gsl_vector_free(v);
		
		}
		//-- version 2 end
	}else if(shyper->version == 3 && iter == 0){
		//-- version 3 begin
		//if (iter>0)continue;

		gsl_vector* gamma = gsl_vector_alloc(m_mu->size2);
		//double unit = 1.0/m_mu->size2;
		int item_num;
		for (size_t t = 0; t < m_mu->size2; t++){
			item_num = items->m_vec_len[t];
			//gsl_vector_set(gamma,t,0);
			gsl_vector_set(gamma,t,item_num/900.0);//for lastfm
			//gsl_vector_set(gamma,t,item_num/900.0);
			//gsl_vector_set(gamma,t,item_num/900.0);
			//gsl_vector_set(gamma,t,item_num/1200.0);//for epinions
		}	
		int iteration_n = shyper->iteration_n;
		//update_mu_SR(users,X, T, B, social_relation,gamma, shyper, iteration_n);
		update_mu_SR_nologistic(users,X, T, B, social_relation,gamma, shyper, iteration_n);
		gsl_vector_free(gamma);
		//-- version 3 end

	}



	gsl_matrix_free(P);
	return likelihood;
}

void s_expo::update_mu_SR(const r_data* users, gsl_matrix * X, gsl_matrix * T, gsl_matrix * B,const r_data* social_relation, gsl_vector * gamma, const sexpo_hyperparameter* shyper,int iteration_n){
	// use stochastic gradient descent
	gsl_matrix* m_S  = gsl_matrix_alloc(n_users, n_items);
	gsl_vector* partial  = gsl_vector_alloc(X->size2);
	double result;
	for(int i = 0; i < iteration_n; i++){
		// random choose example,
		double tmp,gradient_t,gradient_x,gradient_b;
		int random_u = (int)runiform_int(X->size1);
		int m = social_relation->m_vec_len[random_u];
		while(m == 0){
			random_u = (int)runiform_int(X->size1);
			m = social_relation->m_vec_len[random_u];
				
		}
 
		// updating with exposure recorded 0 or 1....begin...
		int n = users->m_vec_len[random_u];	
        	//int* item_ids = users->m_vec_data[random_u];
		int random_i = (int)runiform_int(n);
		//int ui_value = item_ids[random_i];

	 	// end....
       		int* user_ids = social_relation->m_vec_data[random_u];
		double gamma_i = gsl_vector_get(gamma,random_i);
       		if(m > 0){
        		int k_id = (int)runiform_int(m); 
        		int random_k = user_ids[k_id];
        

			gsl_vector_view x = gsl_matrix_row(X, random_u);
			gsl_vector_view t = gsl_matrix_row(T, random_i);
			gsl_vector_view b = gsl_matrix_row(B, random_k);
			// logistic differential equation: g'(x) = g(x)g(1-x)
				
			// update T_{i}
			gsl_blas_ddot(&x.vector, &t.vector, &result);
			tmp = logistic(result + gamma_i);
			//gradient_t = tmp *logistic(1-result) * (tmp - ui_value);
			gradient_t = tmp *logistic(1-result) * (tmp - gsl_matrix_get(m_mu, random_u, random_i));
			//gsl_vector_memcpy(&partial_t.vector, &t.vector);
			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_t),&t.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate * gradient_t, &x.vector, &t.vector);


			//update B_{k}
			gsl_blas_ddot(&x.vector, &b.vector, &result);
			tmp = logistic(result);
			gradient_b = tmp *logistic(1-result) * (tmp - gsl_matrix_get(m_S, random_u, random_k));
			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_b),&b.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate * gradient_b * shyper->lam_sr, &x.vector, &b.vector);

			//update X_{u}
			gsl_blas_ddot(&x.vector, &t.vector, &result);
			tmp = logistic(result + gamma_i);
			//gradient_x = tmp *logistic(1-result) * (tmp - ui_value);
			gradient_x = tmp *logistic(1-result) * (tmp - gsl_matrix_get(m_mu, random_u, random_i));
			gsl_vector_memcpy(partial, &t.vector);
			gsl_blas_dscal(gradient_x ,partial);

			gsl_blas_ddot(&x.vector, &b.vector, &result);
			tmp = logistic(result);
			gradient_b = tmp *logistic(1-result) * (tmp - gsl_matrix_get(m_S, random_u, random_k));
			gsl_blas_daxpy(gradient_b * shyper->lam_sr, &b.vector, partial);

			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_x),&x.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate, partial, &x.vector);
		}else{
			// this user u has no friends
			gsl_vector_view x = gsl_matrix_row(X, random_u);
			gsl_vector_view t = gsl_matrix_row(T, random_i);

			// update T_{i}
			gsl_blas_ddot(&x.vector, &t.vector, &result);
			tmp = logistic(result + gamma_i);
			gradient_t = tmp *logistic(1-result) * (tmp - gsl_matrix_get(m_mu, random_u, random_i));
			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_t),&t.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate * gradient_t, &x.vector, &t.vector);

			//update X_{u}
			gsl_blas_ddot(&x.vector, &t.vector, &result);
			tmp = logistic(result + gamma_i);
			gradient_x = tmp *logistic(1-result) * (tmp - gsl_matrix_get(m_mu, random_u, random_i));
			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_x),&x.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate * gradient_x, &t.vector, &x.vector);

		}
	}
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0,X,T,0.0,m_mu);
	
	for (size_t i = 0; i < m_mu->size1; i++)
    		for (size_t j = 0; j < m_mu->size2; j++){
      			result = gsl_matrix_get(m_mu,i,j);
			//mset(m_mu, i, j, result); 
			mset(m_mu, i, j, logistic(result)); 
	}
	
	gsl_vector_free(partial);
	gsl_matrix_free(m_S);
}

void s_expo::basemodel_SR_noexposure( gsl_matrix * X, gsl_matrix * T, gsl_matrix * B,const r_data* users,  const r_data* social_relation, const sexpo_hyperparameter* shyper,int iteration_n){
	// use stochastic gradient descent
	gsl_matrix* m_S  = gsl_matrix_alloc(n_users, n_items);
	gsl_vector* partial  = gsl_vector_alloc(X->size2);
	double result;

	vector<int> user_has_friends;
	for(int i = 0; i < n_users; i++){
		if(social_relation->m_vec_len[i] != 0){
			user_has_friends.push_back(i);
		}
	}
	for(int i = 0; i < iteration_n; i++){
		// random choose example,

        double tmp,gradient_t,gradient_x,gradient_b, gradient_g;
		int random_u = (int)runiform_int(user_has_friends.size());
		random_u = user_has_friends.at(random_u);
		int m = social_relation->m_vec_len[random_u];
	
		// updating with exposure recorded 0 or 1....begin...
		int n = users->m_vec_len[random_u];	
        	int* item_ids = users->m_vec_data[random_u];
		int uid = (int)runiform_int(n);
		int random_i = item_ids[uid];
		int ui_value = 1;//item_ids[random_i];

	 	// end....
       	int* user_ids = social_relation->m_vec_data[random_u];
		//double gamma_i = gsl_vector_get(gamma,random_i);

        if(m > 0){
        	int k_id = (int)runiform_int(m); 
        	int random_k = user_ids[k_id];
        	 

			gsl_vector_view x = gsl_matrix_row(X, random_u);
			gsl_vector_view t = gsl_matrix_row(T, random_i);
			gsl_vector_view b = gsl_matrix_row(B, random_k);
			// logistic differential equation: g'(x) = g(x)g(1-x)

			// update T_{i}
			gsl_blas_ddot(&x.vector, &t.vector, &result);
			
			gradient_t = (result  - ui_value);
			//gsl_vector_memcpy(&partial_t.vector, &t.vector);
			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_t),&t.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate * gradient_t, &x.vector, &t.vector);


			//update B_{k}
			gsl_blas_ddot(&x.vector, &b.vector, &result);
			//tmp = logistic(result);
			gradient_b = (result- ui_value);//tmp *logistic(1-result) * (tmp - gsl_matrix_get(m_S, random_u, random_k));
			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_b),&b.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate * gradient_b * shyper->lam_sr, &x.vector, &b.vector);

			//update X_{u}
			gsl_blas_ddot(&x.vector, &t.vector, &result);
			//tmp = logistic(result);
			gradient_x = (result - ui_value);
			gsl_vector_memcpy(partial, &t.vector);
			gsl_blas_dscal(gradient_x ,partial);

			gsl_blas_ddot(&x.vector, &b.vector, &result);
			//tmp = logistic(result);
			gradient_b = (result  - ui_value);
			gsl_blas_daxpy(gradient_b * shyper->lam_sr, &b.vector, partial);

			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_x),&x.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate, partial, &x.vector);
		}else{
			// this user u has no friends
			gsl_vector_view x = gsl_matrix_row(X, random_u);
			gsl_vector_view t = gsl_matrix_row(T, random_i);

			// update T_{i}
			gsl_blas_ddot(&x.vector, &t.vector, &result);
			//tmp = logistic(result);
			gradient_t =(result - ui_value);
			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_t),&t.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate * gradient_t, &x.vector, &t.vector);

			//update X_{u}
			gsl_blas_ddot(&x.vector, &t.vector, &result);
			//tmp = logistic(result);
			gradient_x = (result - ui_value);
			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_x),&x.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate * gradient_x, &t.vector, &x.vector);

		}
	}

	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0,X,T,0.0,m_mu);

	for (size_t i = 0; i < m_mu->size1; i++)
    		for (size_t j = 0; j < m_mu->size2; j++){
      			result = gsl_matrix_get(m_mu,i,j);
			mset(m_mu, i, j, logistic(result)); 
	}

	gsl_vector_free(partial);
	gsl_matrix_free(m_S);
}

void s_expo::fit_basemodel_sr(const r_data* users, const r_data* items, const r_data* social, 
	const r_data* vad_data, const sexpo_hyperparameter* shyper){
	//init_para(int users, int items, shyper) //TO DO
	//update()
	omp_set_num_threads(20);


	time_t start, current;
	time(&start);
	int elapsed = 0;

	gsl_matrix * Xb = gsl_matrix_alloc(m_theta->size1, m_theta->size2);


	for (size_t i = 0; i < Xb->size1; i++)
    	for (size_t j = 0; j < Xb->size2; j++)
      		mset(Xb, i, j,runiform());

  	basemodel_SR_noexposure( m_theta, m_beta, Xb, users, social, shyper,shyper->iteration_n);

  	time(&current);
	elapsed = (int)difftime(current,start);
	printf("toal time = %06d, \n", elapsed);

	gsl_matrix_free(Xb);


}

void s_expo::update_mu_SR_nologistic(const r_data* users, gsl_matrix * X, gsl_matrix * T, gsl_matrix * B,const r_data* social_relation, gsl_vector * gamma, const sexpo_hyperparameter* shyper,int iteration_n){
	// use stochastic gradient descent
	//gsl_matrix* m_S  = gsl_matrix_alloc(n_users, n_items);
	gsl_vector* partial  = gsl_vector_alloc(X->size2);
	double result;

	vector<int> user_has_friends;
	for(int i = 0; i < n_users; i++){
		if(social_relation->m_vec_len[i] != 0){
			user_has_friends.push_back(i);
		}
	}
	for(int i = 0; i < iteration_n; i++){
		// random choose example,
		double tmp,gradient_t,gradient_x,gradient_b, gradient_g;
		int random_u = (int)runiform_int(user_has_friends.size());
		random_u = user_has_friends.at(random_u);
		int m = social_relation->m_vec_len[random_u];
	
		// updating with exposure recorded 0 or 1....begin...
		int n = users->m_vec_len[random_u];	
        	int* item_ids = users->m_vec_data[random_u];
		int uid = (int)runiform_int(n);
		int random_i = item_ids[uid];
		int ui_value = 1;

	 	// end....
       	int* user_ids = social_relation->m_vec_data[random_u];
		double gamma_i = gsl_vector_get(gamma,random_i);
       	if(m > 0){
        	int k_id = (int)runiform_int(m); 
        	int random_k = user_ids[k_id];
        

			gsl_vector_view x = gsl_matrix_row(X, random_u);
			gsl_vector_view t = gsl_matrix_row(T, random_i);
			gsl_vector_view b = gsl_matrix_row(B, random_k);
			// logistic differential equation: g'(x) = g(x)g(1-x)
				
			// update T_{i}
			gsl_blas_ddot(&x.vector, &t.vector, &result);
			//tmp = logistic(result + gamma_i);
			//gradient_t = tmp *logistic(1-result) * (tmp - ui_value);
			gradient_t = (result + gamma_i - ui_value);
			//gsl_vector_memcpy(&partial_t.vector, &t.vector);
			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_t),&t.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate * gradient_t, &x.vector, &t.vector);

		
			//update B_{k}
			gsl_blas_ddot(&x.vector, &b.vector, &result);
			//tmp = logistic(result);
			//gradient_b = tmp *logistic(1-result) * (tmp - gsl_matrix_get(m_S, random_u, random_k));
			gradient_b = (result + gamma_i - ui_value);
			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_b),&b.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate * gradient_b * shyper->lam_sr, &x.vector, &b.vector);

			//update X_{u}
			gsl_blas_ddot(&x.vector, &t.vector, &result);
			//tmp = logistic(result + gamma_i);
			//gradient_x = tmp *logistic(1-result) * (tmp - ui_value);
			gradient_x = (result + gamma_i - ui_value);//tmp *logistic(1-result) * (tmp - gsl_matrix_get(m_mu, random_u, random_i));
			gsl_vector_memcpy(partial, &t.vector);
			gsl_blas_dscal(gradient_x ,partial);

			gsl_blas_ddot(&x.vector, &b.vector, &result);
			//tmp = logistic(result);
			gradient_b = (result + gamma_i - ui_value);//tmp *logistic(1-result) * (tmp - gsl_matrix_get(m_S, random_u, random_k));
			gsl_blas_daxpy(gradient_b * shyper->lam_sr, &b.vector, partial);

			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_x),&x.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate, partial, &x.vector);
			
			// update gamma_i				
			gsl_blas_ddot(&x.vector, &t.vector, &result);
			gradient_g = result + gamma_i -ui_value;
			gamma_i = (1 - shyper->learning_rate* shyper->lam_t)*gamma_i - shyper->learning_rate * gradient_g;		
			gsl_vector_set(gamma,random_i,gamma_i);
		}else{
			// this user u has no friends
			gsl_vector_view x = gsl_matrix_row(X, random_u);
			gsl_vector_view t = gsl_matrix_row(T, random_i);

			// update T_{i}
			gsl_blas_ddot(&x.vector, &t.vector, &result);
			tmp = logistic(result + gamma_i);
			gradient_t = tmp *logistic(1-result) * (tmp - gsl_matrix_get(m_mu, random_u, random_i));
			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_t),&t.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate * gradient_t, &x.vector, &t.vector);

			//update X_{u}
			gsl_blas_ddot(&x.vector, &t.vector, &result);
			tmp = logistic(result + gamma_i);
			gradient_x = tmp *logistic(1-result) * (tmp - gsl_matrix_get(m_mu, random_u, random_i));
			gsl_blas_dscal((1- shyper->learning_rate * shyper->lam_x),&x.vector);
			gsl_blas_daxpy(-1 * shyper->learning_rate * gradient_x, &t.vector, &x.vector);

		}
	}
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0,X,T,0.0,m_mu);
	
	for (size_t i = 0; i < m_mu->size1; i++)
    		for (size_t j = 0; j < m_mu->size2; j++){
      			result = gsl_matrix_get(m_mu,i,j);
			mset(m_mu, i, j, result + gsl_vector_get(gamma,j)); 
	}
	
	gsl_vector_free(partial);
	//gsl_matrix_free(m_S);
}
