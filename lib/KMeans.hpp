//
//  KMeans.hpp
//  K-means Clustering
//
//  @author Pei Xu
//  @version 1.0 11/19/2016
//

#ifndef KMeans_hpp
#define KMeans_hpp

#include <deque>
#include <set>
#include <unordered_map>
#include <ostream>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>

#include "Eigen/Core"
#include "Eigen/Sparse"

class KMeans {
 
public:
    // If the random seed is equal to this flag, a new seed will be generated for initializing centroids.
    static int UNASSIGNED_RANDOM_SEED_FLAG;
    
    // A raw structure of data points
    struct Point
    {
        int id;
        std::deque<int> attribute;  // It is better to use a decreasing order
        std::deque<double> value;
        Point(const int & id, const std::deque<int> & attribute, const std::deque<double> & value) : id(id), attribute(attribute), value(value) {}
    };
    
    // Stream for displaying working log when clustering
    std::ostream * log_stream;
    
    // Random seed for generating initial centroids
    int seed;
    
private:
    
    // The numerical structure of data points
    struct _S_Point
    {
        int id;
        Eigen::SparseVector<double> vec;
        double l2norm;
        int centroid = -1;       // id of the centorid as well as the id of its cluster. See the stucture of _Centroid
        _S_Point(int id) : id(id) {}
    };
    
    // A combination structure of Centroid and Cluster
    struct _Centroid
    {
        int id;
        std::deque<_S_Point *> pts;
        Eigen::VectorXd vec;
        double l2norm;
        _Centroid(const int & id, const Eigen::VectorXd & vec, const double & l2norm) : id(id), vec(vec), l2norm(l2norm) {}
    };
    
    // Number of clusters expected
    int _n_clusters;
    // Number of data points
    int _n_points = 0;
    // Threshold for stop ceriterion.
    // K-means iteration will keep going if the number of centroids being updated is greater than this threshold
    int _update_threshold = 0;
    // Dimension of documents
    int _dim = -1;
    // Table of points
    std::unordered_map<int, Point *> _pts;
    // List of structured points
    std::deque<_S_Point *> _s_pts;
    // List of centroids
    std::deque<_Centroid *> _centroids;
    // Flag for multiple runs
    bool _completed = false;
    // Value of the objective function.
    // It is the summation of distance/dissimilarity between a point and the centroid of its cluster.
    // Our goal is to minimize this value.
    double _obj_value;  // this value would be updated after assigning points (_assignPoints() function)
    // Number of centroids changed, value of objetive function and time used at each iteration
    // from the 1st to last iteration.
    std::vector<std::tuple<int, double, double> > _iter_info;
    // Total time taken
    double _total_time_taken = 0;
    // Clustering solution
    std::deque<std::deque<int> > _clustering;    // each element is a set of a cluster's points
                                                 // the key value is the id of the cluster
    std::unordered_map<int, int> _point_clustering; // <point.id, cluster.id>
public:
    KMeans(const int & n_cluster);
    ~KMeans();
    // Set the number of clusters
    void setNumberOfClusters(const int & n_cluster);
    // Set a threshold for K-means strop cerition.
    // K-means iteration will keep going if the number of centroids being updated is greater than this threshold
    void setCentroidUpdateThreshold(const int & threshold);
    // Set the stream for outputing log information
    void setLogStream(std::ostream * log_stream);
    // Set the random seed for generating initial centroids
    void setRandomSeed(const int & seed);
    // Add a data object
    int addDataPoint(const int & id, const std::deque<int> & attribute, const std::deque<double> & value);
    // Run clustering
    int run();
    // Perform evaluation
    void evaluate(const std::unordered_map<std::string, std::set<int> > & class_points_map);
    // Get the total time taken of clustering in the unit of second
    const double & getTimeElapse();
    // Get the objective function's current value
    // Normally it should be the final value of the last clustering action.
    const double & getObjValue();
    // Get the number of clusters whose centriods are updated, the value of objective function and
    // the time taken at each iteration
    const std::vector<std::tuple<int, double, double> > & getIterationInfo();
    // Get a map where the key is a data object's id and the value is the id of its cluster
    const std::unordered_map<int, int> & getEachPointCluster();
    // Get a list of clusters
    // where the order of each element is the id of the correpsonding cluster and
    // the value of each element is a list of the ids of all data objects who belong to the cluster
    const std::deque<std::deque<int> > & getClusters();
    // Output log information
    void log(const std::string & message);
private:
    void _vectorizeData();
    std::deque<int> _initializeCentroids();
    int _assignPoints();
    int _updateCentroids(const std::set<int> & centroid_ids);
};

#endif /* KMeans_hpp */
