// class for eval

#ifndef EVAL_H
#define EVAL_H

#include "utils.h"
#include "data.h"

class eval
{
public:
	eval();
	~eval();
	void set_parameters(char* directory,int num_factors,int num_users,int num_items);
	double mtx_mae(gsl_matrix* m_U,gsl_matrix* m_V, const r_data* test_users,int num_test_users);
	double mtx_rsme(gsl_matrix* m_U, gsl_matrix* m_V, const r_data* test_users,int num_test_users);
	double cal_mae();
	double cal_rsme();

	//
	double ndcg_at_k(const gsl_matrix* m_U,const gsl_matrix* m_V, const r_data* train_user,
	const r_data* vali_user, const r_data* test_user,int num_users, int k);
	double recall_at_k(const gsl_matrix* m_U,const gsl_matrix* m_V, const r_data* train_user,
		const r_data* vali_user, const r_data* test_user,int num_users, int k);
	double recall_at_k_withmu(const gsl_matrix* m_MU, const gsl_matrix* m_U,const gsl_matrix* m_V, const r_data* train_user,
			const r_data* vali_user, const r_data* test_user,int num_users, int k);
	double map_at_k(const gsl_matrix* m_U,const gsl_matrix* m_V, const r_data* train_user,
		const r_data* vali_user, const r_data* test_user,int num_users, int k);
public:
	int num_factors;
	int num_users;
	int num_items;
	double best_mae;
	double best_rsme;
	char mae_path[100];
	char rsme_path[100];
	
	//new metrics
	double best_recall;
	double best_ndcg;
	double best_map;

};

#endif
