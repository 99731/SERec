#include <assert.h>
#include <iostream>
#include <stdio.h>
#include "data.h"

using namespace std;

/**
 * rating matrix
 * each record is a person, and each array contains items.
 */
r_data::r_data() {
}

r_data::~r_data() {
  for (size_t i = 0; i < m_vec_data.size(); i ++) {
    int* ids = m_vec_data[i];
    if (ids != NULL) delete [] ids;
    double* scores = m_vec_score[i];
    if(scores != NULL) delete[] scores;
  }
  m_vec_data.clear();
  m_vec_len.clear();
  m_vec_score.clear();
}

void r_data::read_data(const char * data_filename, int OFFSET) {

  int length = 0, n = 0, id = 0, total = 0;
  double score = 0;
  //int uid = -1, iid = -1,rating = -1;
  FILE * fileptr;
  fileptr = fopen(data_filename, "r");
  
  
  while ((fscanf(fileptr, "%10d", &length) != EOF)) {
    
    int * ids = NULL;
    double * scores = NULL;
    //int r = 0;
    if (length > 0) {
      ids = new int[length];
      scores = new double[length];
      for (n = 0; n < length; n++) {
       	if(2 != fscanf(fileptr, "%10d:%10lf", &id, &score))
        {
		printf("data format %s is wrong!\n",data_filename);
	//	exit(-1);
	}
	 //cout<<"id:\t"<<id<<endl;
        //cout<<"score:\t"<<score<<endl;
        ids[n] = id - OFFSET;
        scores[n] = score;
      }
    }

    m_vec_data.push_back(ids);
    m_vec_len.push_back(length);
    m_vec_score.push_back(scores);
    total += length;
  }
  fclose(fileptr);
  printf("read %d vectors with %d entries ...\n", (int)m_vec_len.size(), total);
}

