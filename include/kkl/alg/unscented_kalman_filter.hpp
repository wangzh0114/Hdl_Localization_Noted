/**
 * UnscentedKalmanFilterX.hpp
 * @author koide
 * 16/02/01
 **/
#ifndef KKL_UNSCENTED_KALMAN_FILTER_X_HPP
#define KKL_UNSCENTED_KALMAN_FILTER_X_HPP

#include <random>
#include <Eigen/Dense>

namespace kkl {
  namespace alg {

/**
 * @brief Unscented Kalman Filter class
 * @param T        scaler type
 * @param System   system class to be estimated
 */
template<typename T, class System>
class UnscentedKalmanFilterX {
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
public:
  /**
   * @brief constructor
   * @param system               system to be estimated
   * @param state_dim            state vector dimension
   * @param input_dim            input vector dimension
   * @param measurement_dim      measurement vector dimension
   * @param process_noise        process noise covariance (state_dim x state_dim)
   * @param measurement_noise    measurement noise covariance (measurement_dim x measuremend_dim)
   * @param mean                 initial mean
   * @param cov                  initial covariance
   */
  UnscentedKalmanFilterX(const System& system, int state_dim, int input_dim, int measurement_dim, const MatrixXt& process_noise, const MatrixXt& measurement_noise, const VectorXt& mean, const MatrixXt& cov)
    : state_dim(state_dim),
    input_dim(input_dim),
    measurement_dim(measurement_dim),
    N(state_dim),
    M(input_dim),
    K(measurement_dim),
    S(2 * state_dim + 1),
    mean(mean),
    cov(cov),
    system(system),
    process_noise(process_noise),
    measurement_noise(measurement_noise),
    lambda(1),
    normal_dist(0.0, 1.0)
  {
    weights.resize(S, 1); // 作者把cov_w, 和mean_w近似认为两个相等， 3.67式
    sigma_points.resize(S, N);
    ext_weights.resize(2 * (N + K) + 1, 1);
    ext_sigma_points.resize(2 * (N + K) + 1, N + K);
    expected_measurements.resize(2 * (N + K) + 1, K);

    // initialize weights for unscented filter
    weights[0] = lambda / (N + lambda);
    for (int i = 1; i < 2 * N + 1; i++) {
      weights[i] = 1 / (2 * (N + lambda));
    }

    // weights for extended state space which includes error variances
    ext_weights[0] = lambda / (N + K + lambda); // 扩展状态包含了measurement
    for (int i = 1; i < 2 * (N + K) + 1; i++) {
      ext_weights[i] = 1 / (2 * (N + K + lambda));
    }
  }

  /**
   * @brief predict
   * @param control  input vector
   */
  void predict() { // 没有imu数据时，用常量速度模型把从estimator状态从上一帧时刻预测到当前帧时刻
    // calculate sigma points
    ensurePositiveFinite(cov);
    computeSigmaPoints(mean, cov, sigma_points);
    // sigma_points更新,用在posesystem中定义的f函数来进行
    for (int i = 0; i < S; i++) {
      sigma_points.row(i) = system.f(sigma_points.row(i)); //《probabilistic robotics》 Table 3.4 line 3.
    }
    /*----至此，sigma_points里存储的就是当前时刻的由ukf输出的系统状态-----*/

    // 过程噪声，即ukf中的矩阵R
    const auto& R = process_noise;

    // unscented transform 定义当前的平均状态和协方差矩阵，并设置为0矩阵
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    // 加权平均，预测状态
    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    // 根据状态预测协方差
    for (int i = 0; i < S; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    // 附加过程噪声R，在pose_estimator中给出初值
    cov_pred += R;
    // 更新mean和cov
    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief predict
   * @param control  input vector
   */
  void predict(const VectorXt& control) { //上一帧laser到当前帧laser之间的所有imu meas, 每来一次imu mea，把estimator的状态做一次predict
    // calculate sigma points
    ensurePositiveFinite(cov);
    computeSigmaPoints(mean, cov, sigma_points);
    // sigma_points更新,用在posesystem中定义的f函数来进行
    for (int i = 0; i < S; i++) {
      sigma_points.row(i) = system.f(sigma_points.row(i), control); // (acc, gyro), 加速度没有去重力分量
    }
    /*----至此，sigma_points里存储的就是当前时刻的由ukf输出的系统状态-----*/

    // 过程噪声，即ukf中的矩阵R
    const auto& R = process_noise;

    // unscented transform 定义当前的平均状态和协方差矩阵，并设置为0矩阵
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    // 加权平均，预测状态
    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    // 根据状态预测协方差
    for (int i = 0; i < S; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    // 附加过程噪声R，在pose_estimator中给出初值
    cov_pred += R;
    // 更新mean和cov
    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief correct
   * @param measurement  measurement vector
   */
  void correct(const VectorXt& measurement) { // ndt在k时刻初值的基础上匹配计算出来k时刻的P，Q
    // N-状态方程维度,K-观测维度
    // create extended state space which includes error variances
    VectorXt ext_mean_pred = VectorXt::Zero(N + K, 1);
    MatrixXt ext_cov_pred = MatrixXt::Zero(N + K, N + K);
    // 左上角N行1列
    ext_mean_pred.topLeftCorner(N, 1) = VectorXt(mean); // mean: 预测k时刻状态的均值
    // 左上角N行N列
    ext_cov_pred.topLeftCorner(N, N) = MatrixXt(cov);   // cov:  预测k时刻状态的cov
    // 右下角K行K列,初始化为在pose_estimator输入的噪声,位置噪声0.01，四元数0.001
    ext_cov_pred.bottomRightCorner(K, K) = measurement_noise;
    // ext_cov_pred.bottomRightCorner(K, K) = measurement_noise; 
    //计算sigma points时不应该加测量噪声，注释掉上面行

    /*------- 经过以上操作，现在扩展状态变量前N项为mean，扩展协方差左上角为N*N的cov，右下角为K*K的观测噪声--------*/

    // 验证并计算
    ensurePositiveFinite(ext_cov_pred);
    // 利用扩展状态空间的参数计算sigma点
    computeSigmaPoints(ext_mean_pred, ext_cov_pred, ext_sigma_points); // line 6. 预测出来k时刻状态的sigma points

    // unscented transform
    // ext_sigma_points、expected_measurements是（2 * (N + K) + 1, K)的矩阵
    expected_measurements.setZero();
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurements.row(i) = system.h(ext_sigma_points.row(i).transpose().topLeftCorner(N, 1));
      // line 7. 用预测出来k时刻sigma points代到测量方程，得到k时刻预测的测量值(sigma points表示）
      expected_measurements.row(i) += VectorXt(ext_sigma_points.row(i).transpose().bottomRightCorner(K, 1));
    }

    // 加权平均,同predict函数相似
    VectorXt expected_measurement_mean = VectorXt::Zero(K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);
    }
    MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += ext_weights[i] * diff * diff.transpose();
    }

    // calculated transformed covariance
    // 转换方差,用于计算sigama，进而计算卡尔曼增益
    MatrixXt sigma = MatrixXt::Zero(N + K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      auto diffA = (ext_sigma_points.row(i).transpose() - ext_mean_pred); // 预测出来k时刻状态的均值 和 预测出来k时刻状态的sigma points 之间的差异
      auto diffB = (expected_measurements.row(i).transpose() - expected_measurement_mean); // 预测出来k时刻测量值的均值 和 预测出来k时刻测量值的sigma points之间的差异
      sigma += ext_weights[i] * (diffA * diffB.transpose());
    }

    kalman_gain = sigma * expected_measurement_cov.inverse(); // line 11.
    const auto& K = kalman_gain;

    // 更新最后的ukf
    VectorXt ext_mean = ext_mean_pred + K * (measurement - expected_measurement_mean);
    MatrixXt ext_cov = ext_cov_pred - K * expected_measurement_cov * K.transpose();

    mean = ext_mean.topLeftCorner(N, 1); // 最终k时刻laser在map下的位姿是由 line 12. 决定
    cov = ext_cov.topLeftCorner(N, N);
  }

  /*			getter			*/
  const VectorXt& getMean() const { return mean; }
  const MatrixXt& getCov() const { return cov; }
  const MatrixXt& getSigmaPoints() const { return sigma_points; }

  System& getSystem() { return system; }
  const System& getSystem() const { return system; }
  const MatrixXt& getProcessNoiseCov() const { return process_noise; }
  const MatrixXt& getMeasurementNoiseCov() const { return measurement_noise; }

  const MatrixXt& getKalmanGain() const { return kalman_gain; }

  /*			setter			*/
  UnscentedKalmanFilterX& setMean(const VectorXt& m) { mean = m;			return *this; }
  UnscentedKalmanFilterX& setCov(const MatrixXt& s) { cov = s;			return *this; }

  UnscentedKalmanFilterX& setProcessNoiseCov(const MatrixXt& p) { process_noise = p;			return *this; }
  UnscentedKalmanFilterX& setMeasurementNoiseCov(const MatrixXt& m) { measurement_noise = m;	return *this; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  const int state_dim;
  const int input_dim;
  const int measurement_dim;

  const int N;
  const int M;
  const int K;
  const int S;

public:
  VectorXt mean;
  MatrixXt cov;

  System system;
  MatrixXt process_noise;		//
  MatrixXt measurement_noise;	//

  T lambda;
  VectorXt weights;

  MatrixXt sigma_points;

  VectorXt ext_weights;
  MatrixXt ext_sigma_points;
  MatrixXt expected_measurements;

private:
  /**
   * @brief compute sigma points
   * @param mean          mean
   * @param cov           covariance
   * @param sigma_points  calculated sigma points
   */
  void computeSigmaPoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& sigma_points) {
    const int n = mean.size();
    assert(cov.rows() == n && cov.cols() == n);

    //llt分解,求Cholesky分解A=LL^*=U^*U,L是下三角矩阵
    Eigen::LLT<MatrixXt> llt;
    llt.compute((n + lambda) * cov);
    MatrixXt l = llt.matrixL();

    // mean是列向量,这里会自动转置处理
    sigma_points.row(0) = mean;
    for (int i = 0; i < n; i++) {
      sigma_points.row(1 + i * 2) = mean + l.col(i); // 奇数1357
      sigma_points.row(1 + i * 2 + 1) = mean - l.col(i); // 偶数2468
    }
  }

  /**
   * @brief make covariance matrix positive finite
   * @param cov  covariance matrix
   */
  void ensurePositiveFinite(MatrixXt& cov) { // 未实际应用
    return;
    const double eps = 1e-9;

    Eigen::EigenSolver<MatrixXt> solver(cov);
    MatrixXt D = solver.pseudoEigenvalueMatrix(); // 特征值
    MatrixXt V = solver.pseudoEigenvectors();     // 特征向量
    for (int i = 0; i < D.rows(); i++) {
      if (D(i, i) < eps) {
        D(i, i) = eps;
      }
    }

    cov = V * D * V.inverse();
  }

public:
  MatrixXt kalman_gain;

  std::mt19937 mt;
  std::normal_distribution<T> normal_dist;
};

  }
}


#endif
