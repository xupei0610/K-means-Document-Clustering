//
//  KMeans.cpp
//  K-means Clustering
//
//  @author Pei Xu
//  @version 1.0 11/19/2016
//

#include "KMeans.hpp"

int KMeans::UNASSIGNED_RANDOM_SEED_FLAG = 0;

KMeans::KMeans(const int & n_clusters)
{
    this->setNumberOfClusters(n_clusters);
    this->seed = KMeans::UNASSIGNED_RANDOM_SEED_FLAG;
}

KMeans::~KMeans()
{
    for(auto p: this->_pts)
        delete p.second;
    for(auto p: this->_s_pts)
        delete p;
    for(auto c: this->_centroids)
        delete c;
}

void KMeans::setNumberOfClusters(const int & n_clusters)
{
    this->_n_clusters = n_clusters;
}

void KMeans::setLogStream(std::ostream * log_stream)
{
    this->log_stream = log_stream;
}

void KMeans::setRandomSeed(const int & seed)
{
    this->seed = seed;
}

void KMeans::setCentroidUpdateThreshold(const int & threshold)
{
    this->_update_threshold = threshold < 0 ? 0 : threshold;
}

const double & KMeans::getTimeElapse()
{
    return this->_total_time_taken;
}

const double & KMeans::getObjValue()
{
    return this->_obj_value;
}

const std::unordered_map<int, int> & KMeans::getEachPointCluster()
{
    return this->_point_clustering;
}

const std::deque<std::deque<int> > & KMeans::getClusters()
{
    return this->_clustering;
}

const std::vector<std::tuple<int, double, double> > & KMeans::getIterationInfo()
{
    return this->_iter_info;
}

int KMeans::addDataPoint(const int & id, const std::deque<int> & attribute, const std::deque<double> & value)
{
    this->_completed = false;
    if (attribute.empty())
        return 1;       // Empty point
    if (attribute.size() != value.size())
        return 2;       // Different number of attributes and values
    if (this->_pts.find(id) != this->_pts.end())
        return 3;       // Repeated point
    // Update the max dimension if needed
    int max_dim = *std::max_element(attribute.begin(), attribute.end());
    if (max_dim > this->_dim)
        this->_dim = max_dim;
    // Record the new point
    this->_pts[id] = new Point(id, attribute, value);
    this->_n_points++;
    return 0;
}

void KMeans::log(const std::string & message)
{
    *this->log_stream << message << std::endl;
}

int KMeans::run()
{
    if (this->_n_points < 1)
        return 0;

    if (this->_n_clusters > this->_n_points)
        this->_n_clusters = this->_n_points;

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

    // Clear old clustering information
    if (this->_completed == true)
    {
        for(auto c: this->_centroids)
            delete c;
        this->_centroids.clear();
        this->_point_clustering.clear();
        this->_clustering.clear();
        this->_iter_info.clear();
        this->_total_time_taken = 0;
        this->_completed = false;
    }
    else
    {
        // Vectorize each point
        this->log("Processing raw data...");
        this->_vectorizeData();
    }

    if (this->_n_clusters > this->_n_points)
        this->_n_clusters = this->_n_points;

    // Generate initial centriods
    this->log("Initialize centroids...");
    this->_initializeCentroids();
//    std::deque<int> inital_centroids = this->_initializeCentroids();
//    std::string init_cens = "  ";
//    for (int i = this->_n_clusters; --i > 0;)
//        init_cens += std::to_string(inital_centroids[i]) + ", ";
//    this->log(init_cens + std::to_string(inital_centroids[0]));

    // Start clustering
    this->log("Begin clustering...");
    int iter = 0;
    int updated_cens = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> time;
    double time_elapse;
    do
    {
        time = std::chrono::high_resolution_clock::now();
        iter++;
        updated_cens = this->_assignPoints();
        time_elapse = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1> > > (std::chrono::high_resolution_clock::now() - time).count();
        *this->log_stream << "  Iteration: "  << iter
            << ". Updated Centroids: " << updated_cens
            << ". Obj. Value: " << std::fixed << this->_obj_value
            << ". Time Taken: " << time_elapse << "s" << std::endl;
        this->_iter_info.push_back(std::make_tuple(updated_cens, this->_obj_value, time_elapse));


    } while (updated_cens > this->_update_threshold);

    // Collect clustering solution
    this->log("Collect clustering solution...");
    std::deque<int> cluster_collection;
    for (auto c : this->_centroids)
    {
        cluster_collection.clear();
        for (auto p : c->pts)
        {
            cluster_collection.push_front(p->id);
            this->_point_clustering[p->id] = c->id;
        }
        this->_clustering.push_front(std::move(cluster_collection));
    }

    // Clustering complete
    this->_total_time_taken = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1> > > (std::chrono::high_resolution_clock::now() - start_time).count();
    // this->log("Clustering completed. Total time taken: " + std::to_string(this->_total_time_taken) + "s.");
    *this->log_stream << "Clustering completed. Total time taken: " << this->_total_time_taken << "s.";
    this->_completed = true;
    return iter;
}

void KMeans::_vectorizeData()
{
    int i;
    _S_Point * sp;
    for (auto p : this->_pts)
    {
        sp = new _S_Point(p.first);
        this->_s_pts.push_front(sp);

        sp->vec = Eigen::SparseVector<double>(this->_dim+1);
        i = p.second->attribute.size();
        for (; --i > -1;)
            sp->vec.insert(p.second->attribute[i]) = p.second->value[i];

        // Now only consine similarity is supported. So only the l2norm is needed
        sp->l2norm = sp->vec.norm();
    }
}

std::deque<int> KMeans::_initializeCentroids()
{
    std::deque<int> inital_centroids;
    std::mt19937 sd;
    if (this->seed == KMeans::UNASSIGNED_RANDOM_SEED_FLAG)
        sd.seed(std::random_device()());
    else
        sd.seed(this->seed);
    std::uniform_int_distribution<int> random_gen(0, this->_s_pts.size());
    int rand_num;
    std::set<int> random_num_generated;
    for (int i = this->_n_clusters; --i>-1;)
    {
        rand_num = random_gen(sd);
        if (random_num_generated.find(rand_num) == random_num_generated.end())
        {
            random_num_generated.insert(rand_num);
            this->_centroids.push_front(new _Centroid(
                                                      i, this->_s_pts[rand_num]->vec, this->_s_pts[rand_num]->l2norm)
                                        );
            inital_centroids.push_front(this->_s_pts[rand_num]->id);
            continue;
        }
        ++i;
    }
    return inital_centroids;
}

int KMeans::_assignPoints()
{
    double dissim;
    double min_dissim;
    std::set<int> updated_centroids;
    int updated = 0;
    _Centroid * tar_centroid = nullptr;
    this->_obj_value = 0;

    for (auto c : this->_centroids)
        c->pts.clear();

    for (auto p : this->_s_pts)
    {
        // Looking for the closest centroid
        // Now only cosine dissimilarity is supported
        // cosine dissimilarity is in the range [0, 2] or [0, 1] if tf-idf is used
        min_dissim = 3;
        for (auto c : this->_centroids)
        {
            dissim = 1 - p->vec.dot(c->vec)/(p->l2norm * c->l2norm);

            if (dissim < min_dissim)
            {
                min_dissim = dissim;
                tar_centroid = c;
            }
            if (dissim <= 3e-16)
                break;
        }
        this->_obj_value += min_dissim;
        tar_centroid->pts.push_front(p);
        if (p->centroid != tar_centroid->id)
        {
            if (p->centroid != -1)
                updated_centroids.insert(p->centroid);
            updated_centroids.insert(tar_centroid->id);
            updated++;
            p->centroid = tar_centroid->id;
        }
    }

    if (updated < this->_update_threshold)
        return updated;
    else
        return this->_updateCentroids(updated_centroids);
}

int KMeans::_updateCentroids(const std::set<int> & centroid_ids)
{
    int updated = 0;
    KMeans::_S_Point * p;
    std::deque<int> empty_clusters;
    std::deque<int> nonempty_clusters;

    for (auto cid: centroid_ids)
    {
        if (this->_centroids[cid]->pts.empty())
            empty_clusters.push_front(cid);
        else
            nonempty_clusters.push_front(cid);
    }
    // Deal with empty clusters
    if (!empty_clusters.empty())
    {
        // The basic idea is to pick points from nonempty clusters
        // In order to reduce computation, we firstly pick points from the clusters who need to update the centroids.
        if (!nonempty_clusters.empty())
        {
            for (auto necid : nonempty_clusters)
            {
                if (empty_clusters.empty())
                    break;
                while (!empty_clusters.empty() && this->_centroids[necid]->pts.size() > 1)
                {
                    auto cid = empty_clusters.front();
                    empty_clusters.pop_front();
                    p = this->_centroids[necid]->pts.front();
                    this->_centroids[necid]->pts.pop_front();
                    p->centroid = cid;
                    this->_centroids[cid]->pts.push_front(p);
                    this->_centroids[cid]->vec = p->vec;
                    this->_centroids[cid]->l2norm = p->l2norm;
                }
            }
        }

        // Not enough nonempty clusters
        if (!empty_clusters.empty())
        {
            for (auto cid : empty_clusters)
            {
                // The amount of calculation can be further reduced by
                // picking points from the clusters who has minimal amount of points.
                // But in order to do so, we need to sort current clusters and
                // would result in an increase in computation as well
                for (auto ne_c : this->_centroids)
                {
                    if (ne_c->pts.size() > 1)
                    {
                        p = ne_c->pts.front();
                        ne_c->pts.pop_front();
                        p->centroid = cid;
                        this->_centroids[cid]->pts.push_front(p);
                        this->_centroids[cid]->vec = p->vec;
                        this->_centroids[cid]->l2norm = p->l2norm;
                        nonempty_clusters.push_front(ne_c->id);
                        break;
                    }
                }
            }
        }
    }

    updated = nonempty_clusters.size();
    if (updated < this->_update_threshold)
        return 0;

    for (auto cid : nonempty_clusters)
    {
        this->_centroids[cid]->vec.setZero();
        for (auto p : this->_centroids[cid]->pts)
            this->_centroids[cid]->vec += p->vec;
        this->_centroids[cid]->vec /= this->_centroids[cid]->pts.size();
        this->_centroids[cid]->l2norm = this->_centroids[cid]->vec.norm();
    }

    return updated;
}

void KMeans::evaluate(const std::unordered_map<std::string, std::set<int> > & class_points_map)
{

    // Purity
    std::deque<double> purity(this->_n_clusters+1, 0);
    // Entropy
    std::deque<double> entropy(this->_n_clusters+1, 0);

    // Arrange the map of points to classes
    std::vector<std::pair<std::string, std::set<int> > > vectorized_cp_map(class_points_map.begin(), class_points_map.end());
    std::sort(vectorized_cp_map.begin(), vectorized_cp_map.end(), [](const std::pair<std::string, std::set<int> > & a, const std::pair<std::string, std::set<int> > & b){
        return a.second.size() > b.second.size();
    });
    int total_class = 0;
    std::deque<std::string> class_collection;
    std::unordered_map<int, int> pc_map;
    for (auto c : vectorized_cp_map)
    {
        class_collection.push_back(c.first);
        for (auto p : c.second)
        {
            pc_map[p] = total_class;
        }
        total_class++;
    }

    // Generate comparation matrix
    // each row is a cluster; each column is a category;
    // each element is the number of points who are in the cluster and who belong to the category
    // unfound points are categoried into the last, extra row
    std::deque<std::deque<int> > comp_mat(this->_n_clusters, std::deque<int>(total_class, 0));
    std::deque<int> ungrouped_points(this->_n_clusters, 0);
    std::unordered_map<int, int>::iterator ptr;
    int max, total_pts;
    double division;
    for (auto c : this->_centroids)
    {
        for (auto p : c->pts)
        {
            ptr = pc_map.find(p->id);
            if (ptr == pc_map.end())
            {
                ungrouped_points[c->id] += 1;
                std::cerr << "EMPTY Cluster" << std::endl;
            }
            else
            {
                comp_mat[c->id][ptr->second] += 1;
            }
        }
        max = 0;
        total_pts = c->pts.size();
        for (int i = total_class; --i>-1;)
        {
            if (comp_mat[c->id][i] != 0)
            {
                if (comp_mat[c->id][i] > max)
                    max = comp_mat[c->id][i];

                division = (double)comp_mat[c->id][i]/total_pts;
                entropy[c->id] -= division*log2(division);
            }
        }
        entropy.back() += entropy[c->id]*total_pts;
        purity[c->id] = (double)max/total_pts;
        purity.back() += max;
    }
    entropy.back() /= (double)this->_n_points;
    purity.back() /= (double)this->_n_points;

    // Output analysis results
    *this->log_stream << "\nClustering Analysis:\n";
    *this->log_stream << "# of clusters: " << this->_n_clusters << ",\t# of data obj.: " << this->_n_points << ",\t# of dims: " << this->_dim+1 << ",\tTime taken: " << this->getTimeElapse() << "s\n\n";
    *this->log_stream << "Cluster\tEntropy \tPurity   " << "\tObj. Value\n";

    *this->log_stream << std::setw(7) << " " << "\t" << std::fixed << entropy.back() << "\t" << purity.back() << "\t" << this->getObjValue() << "\n";
    for (int i = 0; i < this->_n_clusters; i++)
    {
        *this->log_stream << std::setw(7) << i << ": " << std::fixed << entropy[i] << "\t" << purity[i] << "\n";
    }

    *this->log_stream << "\nClustering matrix:\n";

    *this->log_stream << "Cluster" << "\t";
    for (auto c : class_collection)
    {
        *this->log_stream << c << "\t";
    }
    *this->log_stream << "\n";

    for (int i = 0; i < this->_n_clusters; i++)
    {
        *this->log_stream << std::setw(2) << i << "\t";
        for (int j = 0; j < total_class; j++)
        {
            *this->log_stream << comp_mat[i][j] << "\t ";
        }
        *this->log_stream << "\n";
    }

    *this->log_stream << std::endl;


}
