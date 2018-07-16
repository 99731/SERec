#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include "socialexpo.h"
#include "eval.h"

gsl_rng * RANDOM_NUMBER = NULL;
int n_threads = 30;

void print_usage_and_exit() {
  // print usage information
  printf("usage:\n");
  printf("      ctr [options]\n");
  printf("      --help:           print help information\n");

  printf("\n");
  printf("      --directory:      save directory, required\n");

  printf("\n");
  printf("      --user:           user file, required\n");
  printf("      --item:           item file, required\n");
  printf("      --a:              positive item weight, default 1\n");
  printf("      --b:              negative item weight, default 0.01 (b < a)\n");
  printf("      --s:              social effects(>=1), default 1.5\n");
  printf("      --lambda_theta:       user vector regularizer, default 0.01\n");
  printf("      --lambda_beta:       item vector regularizer, default 100\n");
  printf("      --learning_rate:  stochastic version for large datasets, default -1. Stochastic learning will be called when > 0.\n");
  printf("\n");

  printf("      --random_seed:    the random seed, default from the current time\n");
  printf("      --save_lag:       the saving lag, default 20 (-1 means no savings for intermediate results)\n");
  printf("      --max_iter:       the max number of iterations, default 200\n");
  printf("\n");


  printf("      --vali_data:      true rate file,required\n");
  printf("      --test_data:      true rate file,required\n");
  printf("      --theta_opt:      optimize theta or not, optional, default not\n");

  printf("*******************************************************************************************************\n");

  exit(0);
}

int main(int argc, char* argv[]){
	
    const char* const short_options = "hd:x:i:a:b:u:v:r:s:m:k:t:e:y:z:w:";
    const struct option long_options[] = {
       {"help",          no_argument,       NULL, 'h'},
       {"directory",     required_argument, NULL, 'd'},
       {"user",          required_argument, NULL, 'x'},
       {"item",          required_argument, NULL, 'i'},
       {"a",             required_argument, NULL, 'a'},
       {"b",             required_argument, NULL, 'b'},
       {"lambda_theta",      required_argument, NULL, 'u'},
       {"lambda_beta",    required_argument, NULL, 'v'},
       {"lambda_mu",     required_argument, NULL, 'r'},
       {"s",      required_argument, NULL, 's'},
       {"max_iter",      required_argument, NULL, 'm'},
       {"n_components",   required_argument, NULL, 'k'},
       {"lambda_y",          required_argument, NULL, 't'},
       {"metric_k",    required_argument, NULL, 'e'},
       {"version",     required_argument, NULL, 'y'},
       {"vali_data", required_argument, NULL, 'z'},
       {"social",  required_argument, NULL, 'w'},
       {"test_data", required_argument, NULL, 'l'},
       {NULL, 0, NULL, 0}};
	  
    char*  user_path = NULL;
    char*  item_path = NULL;
    char*  vali_path = NULL;
    char*  test_path = NULL;
    char*  social_path = NULL;
    char*  directory =  NULL; 
	
	int n_components = 10;
	int max_iter = 5;
	int batch_size = 1000;
	int batch_sgd = 0.1;
	int max_epoch = 10;
	double init_std = 0.01;
	int n_jobs = 20;
	int random_state = 0;
	bool save_params = true;
	//char * save_dir;
	bool early_stopping = true;
	bool verbose = false;
	
	double lam_theta = 1e-5;
	double lam_beta = 1e-5;
	double lam_mu = 0.01;
	double lam_y = 0.01;
	double learning_rate = 0.0001;
	double init_mu = 0.01;
	double a = 1.0;
	double b = 1.0;
	double s = 1.5;
	
	double lam_x = 1;
	double lam_t = 1;
	double lam_b = 1;
	double lam_sr = 10;
	int sr_k = 40;
	int iteration_n = 20000000;

	int metric_k = 50;
	//char filename[500];
	
	int version = 1;
	int cc = 0;
    while(true) {
    	cc = getopt_long(argc, argv, short_options, long_options, NULL);
    	switch(cc) {
    	  case 'h':
	        print_usage_and_exit();
	        break;
	    
	  case 'd':
	        directory = optarg;
	        break;
	
	      case 'x':
	        user_path = optarg;
	        break;
	      case 'i':
	        item_path = optarg;
	        break;
	      case 'a':
	        a = atof(optarg);
	        break;
	      case 'b':
	        b = atof(optarg);
	        break;
	      case 'u':
	        lam_theta = atof(optarg);
	        break;
	      case 'v':
	        lam_beta = atof(optarg);
	        break;
	      case 'z':
	        vali_path = optarg;
	        break;
	      case 'w':
	        social_path = optarg;
	        break;
	      case 'r':
	        lam_mu = atof(optarg);
	        break;
	      case 's':
	        s = atof(optarg);
	        break;
	      case 'm':
	        max_iter =  atoi(optarg);
	        break;
	      case 'k':
	        n_components = atoi(optarg);
	        break;
	      case 't':
	        lam_y = atof(optarg);
	        break;
	      case 'e':
	        metric_k = atoi(optarg);
	        break;
	      case 'y':
	        version = atoi(optarg);
	
	      case 'l':
	        test_path = optarg;
	      case -1:
	        break;
	      case '?':
	        print_usage_and_exit();
	        break;
	      default:
	        break;
    	}
    	if (cc == -1)
     	 break;
  	}

	//save the settings
	sexpo_hyperparameter hyper_para;
	hyper_para.set_train_para(n_components,max_iter,batch_size,batch_sgd, 
		max_epoch, init_std, n_jobs, random_state, save_params, early_stopping, verbose,version);
	hyper_para.set_model_para(lam_theta, lam_beta,lam_mu,lam_y,
		learning_rate, init_mu, a, b, s);
	
	hyper_para.set_SR_para(lam_x, lam_t, lam_b, lam_sr, sr_k,iteration_n);
	printf("the version of mu updating is version %d ...\n",hyper_para.version);
	
	//read users
	printf("reading user matrix from %s ...\n", user_path);
	r_data* users = new r_data();
	users->read_data(user_path);
    int num_users = (int)users->m_vec_data.size();
	
    // read items
    printf("reading item matrix from %s ...\n", item_path);
    r_data* items = new r_data();
    items->read_data(item_path);
    int num_items = (int)items->m_vec_data.size();
	
	//read social file
    printf("reading social matrix from %s ...\n", social_path);
    r_data* social_network = new r_data();
    social_network->read_data(social_path);
    //int num_items = (int)items->m_vec_data.size();

    //read vali_data
    printf("reading vali matrix from %s ...\n", vali_path);
    r_data* vali = new r_data();
    vali->read_data(vali_path);

    //read test_data 
    printf("reading test matrix from %s ...\n", test_path);
    r_data* test = new r_data();
    test->read_data(test_path);

	//create model instance
	RANDOM_NUMBER = new_random_number_generator(3333);
	s_expo* expo = new s_expo();
	printf("users: %d, items: %d\n",num_users,num_items);
	expo->init_para(num_users, num_items, &hyper_para);
	expo->init_para_sr(&hyper_para);
	expo->fit(users, items, social_network, vali, &hyper_para);
	
	//save parameters: theta, beta, mu
	char name[500];
	sprintf(name,"%s/final_theta%c.dat",directory,'0'+hyper_para.version);
	FILE * file_theta = fopen(name,"w");
	mtx_fprintf(file_theta,expo->m_theta);
	fclose(file_theta);

	sprintf(name,"%s/final_beta%c.dat",directory,'0'+hyper_para.version);
	FILE * file_beta = fopen(name,"w");
	mtx_fprintf(file_beta,expo->m_beta);
	fclose(file_beta);

	
	sprintf(name,"%s/final_mu%c.dat",directory, '0'+hyper_para.version);
	FILE * file_mu = fopen(name,"w");
	mtx_fprintf(file_mu,expo->m_mu);
	fclose(file_mu);
	
	eval* evals = new eval();
	printf("eval begin...%ld\n",test->m_vec_data.size());
	evals->set_parameters(directory, n_components, num_users, num_items);
	evals->mtx_mae(expo->m_theta, expo->m_beta, test, test->m_vec_data.size());
	
	printf("recall begin without mu...\n");
	evals->recall_at_k(expo->m_theta, expo->m_beta, users, vali,test, test->m_vec_data.size(),10);
	evals->recall_at_k(expo->m_theta, expo->m_beta, users, vali,test, test->m_vec_data.size(),metric_k);
	
	printf("recall with mu begin...\n");
	evals->recall_at_k_withmu(expo->m_mu, expo->m_theta, expo->m_beta, users, vali,test, test->m_vec_data.size(),10);
	evals->recall_at_k_withmu(expo->m_mu, expo->m_theta, expo->m_beta, users, vali,test, test->m_vec_data.size(),metric_k);
	
	
	printf("map begin...\n");
	evals->map_at_k(expo->m_theta, expo->m_beta, users, vali,test, test->m_vec_data.size(),metric_k);

	printf("ndcg begin...\n");
	evals->ndcg_at_k(expo->m_theta, expo->m_beta, users, vali,test, test->m_vec_data.size(), 10);
	evals->ndcg_at_k(expo->m_theta, expo->m_beta, users, vali,test, test->m_vec_data.size(), metric_k);
	//printf("rsme is %lf\n",evals->best_rsme);
	delete users;
	delete items;
	delete social_network;
	delete vali;
	delete test;
	delete expo;

	return 0;
}
