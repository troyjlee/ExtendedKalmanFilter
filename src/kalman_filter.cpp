#include <iostream>
#include <math.h>
#include "kalman_filter.h"

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_*x_;
  P_ = F_*P_*F_.transpose() + Q_; 
}

void KalmanFilter::Update(const VectorXd &z) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = MatrixXd(2,2);
  S = H_*P_*Ht + R_;
  MatrixXd K = MatrixXd(4,2);
  K = P_*Ht*S.inverse();
  MatrixXd I = MatrixXd::Identity(4, 4);
  P_ = (I - K*H_)*P_;
  x_ = x_ + K*(z-H_*x_);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  float r = sqrt(px*px + py*py);
  if(r < 0.0000001){
    cout << "Error in UpdateEKF: Divide by zero" << endl;
  }
  else{
    VectorXd pred_meas = VectorXd(3);
    pred_meas[0] = r;
    pred_meas[1] = atan2(py,px);
    pred_meas[2] = (px*vx + py*vy)/r;
    VectorXd y = VectorXd(3);
    y = z - pred_meas;
    // difference between angles should be in [-pi, pi]
    if(y(1) > M_PI){
      y(1) -= 2*M_PI;
    } 
    else if(y(1) < -M_PI){
      y(1) += 2*M_PI;
    }
    MatrixXd K = MatrixXd(4,3);
    MatrixXd Ht = H_.transpose();
    K = P_*Ht*(H_*P_*Ht+R_).inverse();
    MatrixXd I = MatrixXd::Identity(4, 4);
    P_ = (I - K*H_)*P_;
    x_ = x_ + K*y;
  }
}
