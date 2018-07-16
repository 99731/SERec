// class for reading the sparse matrix data
// for both user matrix and item matrix
// user matrix:
// number_of_items item1 item2 ...
// item matrix:
// number_of_users user1 user2 ...

#ifndef DATA_H
#define DATA_H

#include <vector>

using namespace std;

class r_data {
public:
  r_data();
  ~r_data();
  void read_data(const char * data_filename, int OFFSET=0);
public:
  vector<int*> m_vec_data;
  vector<int> m_vec_len;
  vector<double*> m_vec_score;
};

#endif // DATA_H
