#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}

//标定外参的旋转矩阵
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    /*
     imu到相机的R    Imu预积分的旋转  两帧图像间的旋转      imu到相机的R
        Qcb *       Qbk_bk+1 =      Qck_ck+1 *            Qcb
                        ｜｜
                        \ / (四元数转矩阵)
        [Qbk_bk+1]r * Qcb  = [Qck_ck+1]l * Qcb
            R                    L
        即：(L - R) * Qcb = 0
        即：A * Rci = 0 --> Ric = Rci.inverse()

        四元数转矩阵：
        [Q]L = Qw*I + [0  -QvT ]
                      [Qv [Qv]x]
        [Q]R = Qw*I + [0  -QvT  ]
                      [Qv -[Qv]x]
        [Qv]x = [0  -qz qy
                 qz  0  -qx
                -qy  qx  0]
    */
    frame_count++;
    // 根据特征关联求解两个连续帧相机的旋转R12
    Rc.push_back(solveRelativeR(corres));//帧间cam的R，由对极几何得到
    // IMU预积分得到的旋转
    Rimu.push_back(delta_q_imu.toRotationMatrix());//帧间IMU的R，由IMU预积分得到
    // 通过外参把imu的旋转转移到相机坐标系
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);//每帧IMU相对于起始帧IMU的R

    Eigen::MatrixXd A(frame_count * 4, 4);
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++)
    {
        Quaterniond r1(Rc[i]);
        Quaterniond r2(Rc_g[i]);

        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        ROS_DEBUG(
            "%d %f", i, angular_distance);

        // huber核函数做加权
        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        Matrix4d L, R;

        //R_bk+1^bk * R_c^b = R_c^b * R_ck+1^ck
        //[Q1(q_bk+1^bk) - Q2(q_ck+1^ck)] * q_c^b = 0
        //L R 分别为左乘和右乘矩阵
        /**
         *         [Q]L = w*I +  [0  -QvT ]
                                 [Qv [Qv]x]
        */
        double w = Quaterniond(Rc[i]).w();
        Vector3d q = Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        /**
         *         [Q]R = Qw*I + [0  -QvT  ]
                                 [Qv -[Qv]x]
        */
        Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R); // 作用在残差上面
    }

    //svd分解中最小奇异值对应的右奇异向量作为旋转四元数
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Quaterniond estimated_R(x);
    ric = estimated_R.toRotationMatrix().inverse();
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();

    // 至少迭代计算了WINDOW_SIZE次，且R的奇异值大于0.25才认为标定成功
    // 倒数第二个奇异值，因为旋转是3个自由度，
    // 因此检查一下第三个小的奇异值是否足够大，通常需要足够的运动激励才能保证得到没有奇异的解
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
    {
        calib_ric_result = ric;
        return true;
    }
    else
        return false;
}

//根据两帧特征点求解两帧的旋转矩阵
Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    if (corres.size() >= 9)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }

        //求解两帧的本质矩阵
        // 这里用的是相机坐标系，因此这个函数得到的也就是E矩阵
        cv::Mat E = cv::findFundamentalMat(ll, rr);
        cv::Mat_<double> R1, R2, t1, t2;
        
        //本质矩阵svd分解得到4组Rt解
        decomposeE(E, R1, R2, t1, t2);
        // 旋转矩阵的行列式应该是1，这里如果是-1就取反
        if (determinant(R1) + 1.0 < 1e-09)
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }

        //通过三角化得到的正深度选择Rt解
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

        //对R求转置
        // 解出来的是R21
        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j); // 这里转换成R12
        return ans_R_eigen;
    }
    return Matrix3d::Identity();
}

/**
 * @brief 通过三角化来检查R t是否合理
 * 
 * @param[in] l l相机的观测
 * @param[in] r r相机的观测
 * @param[in] R 旋转矩阵
 * @param[in] t 位移
 * @return double
*/
double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud;
    // 其中一帧设置为单位阵
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    // 第二帧就设置为R t对应的位姿
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++)
    {
        // 因为是齐次的，所以要求最后一维等于1
        double normal_factor = pointcloud.col(i).at<float>(3);
        // 得到在各自相机坐标系下的3d坐标
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        // 通过深度是是否大于0来判断是否合理
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    return 1.0 * front_count / pointcloud.cols;
}

// 具体解法参考多视角几何或者SLAM14-P169
void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
