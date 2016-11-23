//
//  main.cpp
//  K-means Document Clustering
//
//  @author Pei Xu
//  @version 1.0 11/19/2016
//
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <deque>
#include <limits>
#include <locale>

#include "lib/KMeans.hpp"

void show_help()
{
    std::cout << "This program is for Project 2 of the course CSci5523 Data Mining.\n\n";
    std::cout << "  This program is aimed to clustering documents using the K-means algorithm. It needs five parametrs to run a clustering. They are listed below in order.\n";
    std::cout << "    (1) input-file: This file is the sparse representation of documents in the dataset. It should use the form of CSV. Each line in it represents a document and contains three elements that are separared by a comma. The first element is an integer that represents the id of a document. The second element is a string composed of a set of numbers that are separated by a comma. Each of the number represents a token the document contains. The third element has the same form with the second one, while each of the number in it represents the occurances of the corresponding token in the document. Attention: The same tokens different documents share should have the same representation number.\n";
    std::cout << "    (2) class-file: This file records the actual classification of each document and is used for clustering evaluation. Each line represents a document and contains two elements separated by a comma. The first element is the id of the document and the second element is the name of the actual category to which the document belongs. No evaluation will run if no class-file parameter is provided.\n";
    std::cout << "    (3) clusters: the number of clusters that are hoped to obtain.\n";
    std::cout << "    (4) trails: the times the clustering needs to be conducted. Best solution of the multiple clustering results will be obtained. If this parameter is provided, then the program will use odd numbers from 1 as the random seed to generate initial centroids so that when this parameter is provided, you will always get the same clustering solution for the same trails if the input-file does not change. This is not a must-have parameter\n";
    std::cout << "    (5) output-file: This file is the file that contains the clustering result. Each line of the file has two integer elements. The first element is the id of a document, while the second element is the id of the cluster into which the document is assigned.\n\n";
    std::cout << "  Run data.py under python 3 environment to obtain a set of input and class files from the reuters21578 dataset, while each input file has the extension '.csv' and the class file has the extensin '.class' and each of the extracted tokens are in the file whose extension is '.clabel'. In the '.clabel' files, each line is a token and the line number is the number that represents the token. E.G. if the 5th line is 'abc', then the number that represents the word 'abc' in the '.csv' file is 5.\n" << std::endl;
}

int load_data_file(const char * data_file, KMeans * cluster)
{
    // Open data file
    std::ifstream       file_handle(data_file);
    if (file_handle.fail())
    {
        std::cerr << "Program Stopped." << std::endl;
        std::cerr << "Error: Unable to Open the Input File. " << data_file << std::endl;
        throw;
    }

    // Get file contents
    std::string contents;
    contents.resize(file_handle.seekg(0, std::ios::end).tellg());
    file_handle.seekg(0, std::ios::beg).read(&contents[0], static_cast<std::streamsize>(contents.size()));
    file_handle.close();

    // Parse contents
    std::istringstream cont_stream(contents);
    std::string entry;
    size_t start, end;
    size_t sub_start, sub_end;
    std::string substring;
    int result, id;
    int added = 0;
    std::deque<int> attribute;
    std::deque<double> value;
    auto remove_space = []( char c ) { return std::isspace<char>( c, std::locale::classic() ); };
    while (std::getline(cont_stream, entry))
    {
        entry.erase(std::remove_if(
            entry.begin(), entry.end(), remove_space
        ), entry.end());

        // Each entry is in the following format
        //      id,"token1,token2","frequency1,frequency2"
        entry.pop_back();

        // Obtain the document id
        end = entry.find_first_of(',');
        if (end == std::string::npos)      // Empty line
            continue;

        id = std::atoi(entry.substr(0, end).c_str());

        // Obtain the tokens
        start = end+2;
        end = entry.find_first_of('"', start);
        if (end != std::string::npos)
        {
            substring = entry.substr(start, end-start);
            sub_start = 0;
            sub_end   = substring.find_first_of(',');
            attribute.clear();
            while (sub_end <= std::string::npos)
            {
                attribute.push_front( std::atoi(substring.substr(sub_start, sub_end).c_str()) );
                if (sub_end == std::string::npos)
                    break;
                sub_start = sub_end + 1;
                sub_end = substring.find_first_of(',', sub_start);
            }

            // Obtain the frequencies
            substring = entry.substr(end+3);
            sub_start = 0;
            sub_end   = substring.find_first_of(',');
            value.clear();
            if (id == 0)
              std::cout << substring << std::endl;
            while (sub_end <= std::string::npos)
            {
                value.push_front( std::atoi(substring.substr(sub_start, sub_end).c_str()) );
                if (sub_end == std::string::npos)
                    break;
                sub_start = sub_end + 1;
                sub_end = substring.find_first_of(',', sub_start);
            }
            result = cluster->addDataPoint(id, attribute, value);
            switch (result)
            {
                case 0:
                    added++;
                    break;
                case 2:
                    std::cerr << "Ignore invaild Document. ID: " << id << ". Unmatched tokens with frequencies." << std::endl;
                    break;
                case 3:
                    std::cerr << "Ignore invaild Document. ID: " << id << ". Repeated document." << std::endl;
                    break;
                // case 1:
                //    std::cerr << "Ignore invaild Document. ID: " << doc->id << ". No tokens found." << std::endl;
                // Impossible to happen in this case, see below, such error has been filtered out.
            }
        }
        else
        {
            std::cerr << "Ignore invaild Document. ID: " << id << ". No tokens." << std::endl;
        }
    }
    return added;
}

std::unordered_map<std::string, std::set<int> > load_classfication_file(const char * class_file)
{
    std::unordered_map<std::string, std::set<int> > topic_docs_map;

    // Open data file
    std::ifstream file_handle(class_file);
    if (file_handle.fail())
    {
        std::cerr << "Program Stopped." << std::endl;
        std::cerr << "Error: Unable to Open the Input File." << class_file << std::endl;
        throw;
    }

    // Get file contents
    std::string contents;
    contents.resize(file_handle.seekg(0, std::ios::end).tellg());
    file_handle.seekg(0, std::ios::beg).read(&contents[0], static_cast<std::streamsize>(contents.size()));
    file_handle.close();

    // Parse contents
    std::istringstream cont_stream(contents);
    std::string entry, topic;
    std::unordered_map<std::string, std::set<int> >::iterator ptr;
    int id;
    size_t pos;
    auto remove_space = []( char c ) { return std::isspace<char>( c, std::locale::classic() ); };
    while(std::getline(cont_stream, entry))
    {
        entry.erase(std::remove_if(
           entry.begin(), entry.end(), remove_space
        ), entry.end());

        // Obtain the document's id
        pos = entry.find_first_of(',');
        if (pos == std::string::npos)      // Empty line
            continue;
        id = std::atoi(entry.substr(0, pos).c_str());
        topic = entry.substr(pos + 1);

        ptr = topic_docs_map.find(topic);
        if (ptr == topic_docs_map.end())
        {
            topic_docs_map[topic] = std::set<int>({id});
        }
        else
        {
            ptr->second.insert(id);
        }

    }
    return topic_docs_map;
}

int main(int argc, char * argv[])
{
    // See show_help() for the explanation of these parameters
    char * input_file = nullptr;      // dataset file
    char * class_file = nullptr;    // class file for clustering evaluation
    char * output_file = nullptr;   // output file containing the clustering solution
    int n_clusters = -1;            // number of clusters expected
    int n_trails   = -1;            // the times of tries to do clustering.
        // Run multiple times in order to find a better solution.

    std::ios_base::sync_with_stdio(false);  // No plan to use stdio.h

    // Load parameters
    switch (argc)
    {
        case 1:
            // no parameter provided, show help information
            show_help();
            return 0;
        case 4:
            input_file = argv[1];
            n_clusters = std::atoi(argv[2]);
            output_file = argv[3];
            n_trails = 0;
            break;
        case 5:
            input_file = argv[1];
            n_clusters = std::atoi(argv[2]);
            if (n_clusters == 0)
            {
                n_clusters = std::atoi(argv[3]);
                class_file = argv[2];
                n_trails = 0;
            }
            else
            {
                n_trails = std::atoi(argv[3]);
            }
            output_file = argv[4];
            break;
        case 6:
            input_file = argv[1];
            class_file = argv[2];
            n_clusters = std::atoi(argv[3]);
            n_trails   = std::atoi(argv[4]);
            output_file = argv[5];
            break;
        default:
            std::cerr << "Program Stopped." << std::endl;
            std::cerr << "Error: Wrong Number of Parameters were given." << std::endl;
            std::cerr << "Run the program without any paramter to see help information" << std::endl;
            throw ;

    }

    // Assign default values
    if (n_clusters < 2)
        n_clusters = 2;
    if (n_trails < 0)
        n_trails = 1;

    // Load classification files as the preparation of clustering evaluation
    std::unordered_map<std::string, std::set<int> > topic_docs_map;
    if (class_file != nullptr)
        topic_docs_map = load_classfication_file(class_file);

    // Instance the clustering class
    KMeans * cluster = new KMeans(n_clusters);

    // Load document data from the dataset file into the clustering class
    if (load_data_file(input_file, cluster) < 2)
    {
      std::cerr << "Program Stopped." << std::endl;
      std::cerr << "Error: Less than 2 data objects added. Unable to perform clustering." << std::endl;
      throw;
    }

    // Prepare output file
    std::ofstream output(output_file);
    if (output.fail())
    {
        std::cerr << "Program Stopped." << std::endl;
        std::cerr << "Error: Unable to open the output file." << output_file << std::endl;
        throw;
    }

    // Set threshold for centroid updates
    // K-means iteration will keep going
    // if the number of centroids being updated is greater than the threshold
    cluster->setCentroidUpdateThreshold(0);

    // Run multiple times of clustering
    std::deque<int> rand_seeds;
    std::unordered_map<int, int> solution;
    double obj_val = std::numeric_limits<double>::infinity();

    if (n_trails == 0)
    {
        rand_seeds.push_front(KMeans::UNASSIGNED_RANDOM_SEED_FLAG);
        n_trails = 1;
    }
    else
    {
        for (int i = n_trails; --i > -1;)
        {
            rand_seeds.push_front(2*i+1);
        }
    }
    int i = 0;
    for (auto rs : rand_seeds)
    {
        // Set random seed
        cluster->setRandomSeed(rs);
        std::cout << "Parameters:\n  # of Clusters: " << n_clusters << "\n  # of Tails: " << ++i << "/" << n_trails << "\n  Random Seed: ";
        if (rs == KMeans::UNASSIGNED_RANDOM_SEED_FLAG)
          std::cout << "undefined";
        else
          std::cout << rs;
        std::cout  << "\n  Data File: " << input_file << "\n  Class File: " << (class_file == nullptr ? "undefined" : class_file) << "\n  Output File: " << output_file << "\n" << std::endl;


        // Clustering
        cluster->setLogStream(&std::clog);
        cluster->run();

        // Get the value of objective function
        if (obj_val > cluster->getObjValue())
        {
            obj_val = cluster->getObjValue();
            solution = cluster->getEachPointCluster();
        }

        // Evaluation
        if (class_file != nullptr)
        {
            cluster->setLogStream(&std::cout);
            cluster->evaluate(topic_docs_map);
        }

    }

    // Output best clustering result according the requirement of the project
    for (auto d : solution)
        output << d.first << "," << d.second << "\n";
    output.close();

    delete cluster;

    return 0;
}
