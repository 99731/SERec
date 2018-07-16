
#ifndef SOCIALEXPO_H
#define SOCIALEXPO_H
#include "utils.h"
#include "data.h"
#include "math.h"
struct sexpo_hyperparameter
{
	/*
	Parameters
        ---------
        n_components : int
            Number of latent factors
        max_iter : int
            Maximal number of iterations to perform
        batch_size: int
            Batch size to perform parallel update
        batch_sgd: int
            Batch size for SGD when updating exposure factors
        max_epoch: int
            Number of epochs for SGD
        init_std: double
            The latent factors will be initialized as Normal(0, init_std**2)
        n_jobs: int
            Number of parallel jobs to update latent factors
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        save_params: bool
            Whether to save parameters after each iteration
        save_dir: str
            The directory to save the parameters
        early_stopping: bool
            Whether to early stop the training by monitoring performance on
            validation set
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Model hyperparameters
    */
	int n_components;
	int max_iter;
	int batch_size;
	int batch_sgd;
	int max_epoch;
	double init_std;
	int n_jobs;
	int random_state;
	bool save_params;
	char * save_dir;
	bool early_stopping;
	bool verbose;

	double lam_theta;
	double lam_beta;
	double lam_mu;
	double lam_y;
	double sigma_y;
	double learning_rate;
	double init_mu;
	double a;
	double b;
	double s;

	double lam_b;
	double lam_x;
	double lam_t;
	double lam_sr;
	int sr_k;
	int iteration_n;
	int version;

	sexpo_hyperparameter(){
		lam_theta = 1e-5;
		lam_beta = 1e-5;
		//lam_mu = 1.0;
		lam_y = 1.0;
		sigma_y = sqrt(lam_y);
		//learning_rate = lr;
		init_mu = 0.01;
		a = 1.0;
		b = 1.0;
		s = 1.5;

		lam_sr = 1.0;
		lam_x = 0.01;
		lam_t = 0.01;
		lam_b = 0.01;

		sr_k =20;
		iteration_n = 2000;
		version = 1;
	}
	void set_train_para(int comp, int iter, int b_size, int b_sgd, int max_e, double intstd,
		int nj, int rs, bool sp, bool es, bool v,int ver){
		n_components = comp;
		max_iter =iter;
		batch_size =b_size;
		batch_sgd = b_sgd;
		max_epoch = max_e;
		init_std = intstd;
		n_jobs = nj;
		random_state = rs;
		save_params = sp;
		early_stopping =es;
		verbose = v;
		version = ver;
	}
	void set_SR_para(double x, double t, double b,  double sr, int k, int n){
		lam_x = x;
		lam_t = t;
		lam_b = b;
		lam_sr = sr;
		sr_k = k;
		iteration_n = n;
	}
	void set_model_para(double t, double b, double m, double y, double lr, double im, double aa, double bb, double ss){
		lam_theta = t;
		lam_beta = b;
		lam_mu = m;
		lam_y = y;
		learning_rate = lr;
		init_mu = im;
		a = aa;
		b = bb;
		s = ss;
	}	
	int set_s(const char *new_s){
		if (save_dir) delete save_dir;
		save_dir = new char[strlen(new_s)+1];
		strcpy(save_dir,new_s);
		return 0;
	}
};

class s_expo{
public:
	s_expo();
	~s_expo();
	void init_para(int n_users, int n_items, const sexpo_hyperparameter* shyper);
	void init_para_sr( const sexpo_hyperparameter* shyper);
	void fit(const r_data* users, const r_data* items, const r_data* social, const r_data* vad_data, const sexpo_hyperparameter* shyper);
	double update_factors(const r_data* users, const r_data* items,const r_data* social_relation,const sexpo_hyperparameter* shyper, const int iter);
	void update_mu_SR( const r_data* users, gsl_matrix * X, gsl_matrix * T, gsl_matrix * B,const r_data* social_relation, gsl_vector * gamma, const sexpo_hyperparameter* shyper,int iteration_n);
	void update_mu_SR_nologistic( const r_data* users, gsl_matrix * X, gsl_matrix * T, gsl_matrix * B,const r_data* social_relation, gsl_vector * gamma, const sexpo_hyperparameter* shyper,int iteration_n);
	//void update_exposure(const r_data* users, const r_data* items);// purely based on item popularity
	//---base model----
	void fit_basemodel_sr(const r_data* users, const r_data* items, const r_data* social, 
		const r_data* vad_data, const sexpo_hyperparameter* shyper);
	void basemodel_SR_noexposure( gsl_matrix * X, gsl_matrix * T, gsl_matrix * B,const r_data* users,  const r_data* social_relation, const sexpo_hyperparameter* shyper,int iteration_n);
public:
	gsl_matrix* m_beta;
	gsl_matrix* m_theta;
	gsl_matrix* m_mu;
	gsl_matrix* X;
	gsl_matrix* T;
	gsl_matrix* B;
	int n_users;
	int n_items;
};
#endif
