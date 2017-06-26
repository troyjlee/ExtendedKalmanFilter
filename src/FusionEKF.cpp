#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  // acceleration noise
  A_ = MatrixXd(2,2);
  A_ << 9,0,
        0,9;

  // laser measurement matrix
  H_laser_ << 1,0,0,0,
              0,1,0,0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    cout << "EKF: " << endl;
    // initialize x vector
    // whether lidar or radar, vx and vy will be initialized to 0
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 0, 0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float rho_dot = measurement_pack.raw_measurements_[2];
      ekf_.x_(0) = rho*cos(phi);
      ekf_.x_(1) = rho*sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_(0) = measurement_pack.raw_measurements_[0];
      ekf_.x_(1) = measurement_pack.raw_measurements_[1];
    }
    // initialize time stamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // template for transition matrix
    ekf_.F_ = MatrixXd(4,4);
    ekf_.F_ << 1, 0, 1, 0,
               0, 1, 0, 1,
               0, 0, 1, 0,
               0, 0, 0, 1;


    // initial transition covariance matrix
    ekf_.P_ = MatrixXd(4,4);
    ekf_.P_ << 1,0,0,0,
               0,1,0,0,
               0,0,3,0,
               0,0,0,3;

    // we are now initialized.
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // compute the time elapsed between the current and previous measurements
  long long current_time = measurement_pack.timestamp_;
  // dt - expressed in seconds
  float dt = (current_time - previous_timestamp_) / 1000000.0;	
  previous_timestamp_ = current_time;
	
  // Update F matrix to reflect time elapsed
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;
  // Set the process covariance matrix Q
  MatrixXd G(4,2);
  float dt2 = dt*dt;
  G << dt2/2,0,
       0,dt2/2,
       dt,0,
       0,dt;
  ekf_.Q_ = G*A_*G.transpose();

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
