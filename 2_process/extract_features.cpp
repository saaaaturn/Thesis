//////////////////////////////////////////////////////////////////////////////
//                         PROCESS: features extraction                     //
//Extract node features, adjacency, and statistics from AIG/MIG networks    //
//Output to text files in: matrix/forAIG/<circuit>/<run_tag>/ and           //
//                         matrix/forMIG/<circuit>/<run_tag>/                //
//////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////HEADERS/////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

//Include necessary headers
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
//Include mockturtle and lorina headers for AIG/MIG handling
#include <lorina/aiger.hpp>
#include <mockturtle/io/aiger_reader.hpp>
#include <mockturtle/networks/mig.hpp>
#include <mockturtle/networks/aig.hpp>
//Use normal namespaces in C++
using std::cout;
using std::endl;
using std::string;
//Use mockturtle namespaces for AIG and MIG
using mockturtle::mig_network;
using mockturtle::aig_network;
using lorina::read_aiger;
using lorina::return_code;
using mockturtle::aiger_reader;
namespace fs = std::filesystem;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////FUNCTIONS///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

//1. Extract node features from AIG
void extract_node_features_aig(const aig_network& aig, const string& output_filename) {
    cout << "Opening file: " << output_filename << "\n";
    std::ofstream outfile(output_filename);
    
    if (!outfile.is_open()) {
        cout << "Error: could not open " << output_filename << " for writing\n";
        return;
    }
    
    // Write header: Node_ID Gate_Type Fanin_Count
    outfile << "# Node_ID Gate_Type Fanin_Count\n";
    
    int node_count = 0;
    aig.foreach_node([&](auto node) {
        uint32_t node_id = aig.node_to_index(node);
        int fanin_count = aig.fanin_size(node);
        int gate_type = aig.is_constant(node) ? 0 : 
                       aig.is_pi(node) ? 1 : 2;  // 0=CONST, 1=PI, 2=AND
        
        outfile << node_id << " " << gate_type << " " << fanin_count << "\n";
        node_count++;
    });
    
    outfile.close();
    cout << "✓ Extracted " << node_count << " AIG node features to " << output_filename << "\n";
}

//2. Extract node features from MIG
void extract_node_features_mig(const mig_network& mig, const string& output_filename) {
    std::ofstream outfile(output_filename);
    
    if (!outfile.is_open()) {
        cout << "Error: could not open " << output_filename << " for writing\n";
        return;
    }
    
    // Write header: Node_ID Gate_Type Fanin_Count
    outfile << "# Node_ID Gate_Type Fanin_Count\n";
    
    int node_count = 0;
    mig.foreach_node([&](auto node) {
        uint32_t node_id = mig.node_to_index(node);
        int fanin_count = mig.fanin_size(node);
        int gate_type = mig.is_constant(node) ? 0 : 
                       mig.is_pi(node) ? 1 : 2;  // 0=CONST, 1=PI, 2=MAJ
        
        outfile << node_id << " " << gate_type << " " << fanin_count << "\n";
        node_count++;
    });
    
    outfile.close();
    cout << "Extracted " << node_count << " MIG node features to " << output_filename << "\n";
}

//3. Extract adjacency (edge list) from AIG
void extract_adjacency_aig(const aig_network& aig, const string& output_filename) {
    std::ofstream outfile(output_filename);
    
    if (!outfile.is_open()) {
        cout << "Error: could not open " << output_filename << " for writing\n";
        return;
    }
    
    // Write header: Source Target
    outfile << "# Source Target\n";
    
    int edge_count = 0;
    aig.foreach_node([&](auto target_node) {
        uint32_t target_id = aig.node_to_index(target_node);
        
        // Iterate through all fanins of this node
        aig.foreach_fanin(target_node, [&](auto fanin_signal) {
            auto source_node = aig.get_node(fanin_signal);
            uint32_t source_id = aig.node_to_index(source_node);
            
            outfile << source_id << " " << target_id << "\n";
            edge_count++;
        });
    });
    
    outfile.close();
    cout << "Extracted " << edge_count << " AIG edges to " << output_filename << "\n";
}

//4. Extract adjacency (edge list) from MIG
void extract_adjacency_mig(const mig_network& mig, const string& output_filename) {
    std::ofstream outfile(output_filename);
    
    if (!outfile.is_open()) {
        cout << "Error: could not open " << output_filename << " for writing\n";
        return;
    }
    
    // Write header: Source Target
    outfile << "# Source Target\n";
    
    int edge_count = 0;
    mig.foreach_node([&](auto target_node) {
        uint32_t target_id = mig.node_to_index(target_node);
        
        // Iterate through all fanins of this node
        mig.foreach_fanin(target_node, [&](auto fanin_signal) {
            auto source_node = mig.get_node(fanin_signal);
            uint32_t source_id = mig.node_to_index(source_node);
            
            outfile << source_id << " " << target_id << "\n";
            edge_count++;
        });
    });
    
    outfile.close();
    cout << "Extracted " << edge_count << " MIG edges to " << output_filename << "\n";
}

//5. Extract statistics from circuit
void extract_statistics(const aig_network& aig, const string& output_filename) {
    std::ofstream outfile(output_filename);
    
    if (!outfile.is_open()) {
        cout << "Error: could not open " << output_filename << " for writing\n";
        return;
    }
    
    // Count fanin statistics
    int total_fanin = 0;
    int num_gates = aig.num_gates();
    int num_inv_edges = 0;
    
    aig.foreach_node([&](auto node) {
        if (!aig.is_constant(node) && !aig.is_pi(node)) {
            total_fanin += aig.fanin_size(node);
            aig.foreach_fanin(node, [&](auto fanin_signal) {
                if (aig.is_complemented(fanin_signal)) {
                    ++num_inv_edges;
                }
            });
        }
    });
    
    double avg_fanin = num_gates > 0 ? (double)total_fanin / num_gates : 0.0;
    
    // Write statistics
    outfile << "num_pis " << aig.num_pis() << "\n";
    outfile << "num_pos " << aig.num_pos() << "\n";
    outfile << "num_gates " << num_gates << "\n";
    outfile << "num_inv_edges " << num_inv_edges << "\n";
    outfile << "avg_fanin " << avg_fanin << "\n";
    
    outfile.close();
    cout << "Extracted statistics to " << output_filename << "\n";
}

/////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////MAIN FUNCTION////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    if (argc < 2 || argc > 4) {
        cout << "Error: missing file name\n"
             << "Usage: " << argv[0] << " <filename.aig> [run_tag] [folder_circuit_name]\n";
        return 1;
    }

    string filename = argv[1];
    string run_tag = (argc == 3) ? argv[2] : "";
    if (argc == 4) {
        run_tag = argv[2];
    }

    // Load networks
    aig_network aig;
    mig_network mig;

    cout << "Loading " << filename << " into RAM...\n";
    
    return_code result_aig = read_aiger(filename, aiger_reader(aig));
    if (result_aig != return_code::success) {
        cout << "Error reading " << filename << " into AIG!\n";
        return 1;
    }
    cout << "AIG: " << aig.num_gates() << " gates\n";
    
    return_code result_mig = read_aiger(filename, aiger_reader(mig));
    if (result_mig != return_code::success) {
        cout << "Error reading " << filename << " into MIG!\n";
        return 1;
    }
    cout << "MIG: " << mig.num_gates() << " gates\n";

    // Get matrix file prefix from input filename (without path and extension)
    string basename = fs::path(filename).stem();

    // Optional folder name override so all step matrices can share one run folder.
    string folder_circuit_name = (argc == 4) ? argv[3] : basename;

    // Create output directories.
    fs::path aig_base = fs::path("../matrix/forAIG");
    fs::path mig_base = fs::path("../matrix/forMIG");
    fs::path aig_dir = aig_base;
    fs::path mig_dir = mig_base;
    if (!run_tag.empty()) {
        aig_dir /= folder_circuit_name;
        aig_dir /= run_tag;
        mig_dir /= folder_circuit_name;
        mig_dir /= run_tag;
    }
    fs::create_directories(aig_dir);
    fs::create_directories(mig_dir);
    cout << "AIG matrix output dir: " << aig_dir.string() << "\n";
    cout << "MIG matrix output dir: " << mig_dir.string() << "\n";
    
    // Extract AIG features
    cout << "\n--- Extracting AIG Features ---\n";
    extract_node_features_aig(aig, (aig_dir / (basename + "_node_features.txt")).string());
    extract_adjacency_aig(aig, (aig_dir / (basename + "_adjacency.txt")).string());
    
    // Extract MIG features
    cout << "\n--- Extracting MIG Features ---\n";
    extract_node_features_mig(mig, (mig_dir / (basename + "_node_features.txt")).string());
    extract_adjacency_mig(mig, (mig_dir / (basename + "_adjacency.txt")).string());
    
    // Extract statistics (once, for the circuit)
    cout << "\n--- Extracting Statistics ---\n";
    string stats_file = (aig_dir / (basename + "_statistics.txt")).string();
    extract_statistics(aig, stats_file);
    
    cout << "\n=== Feature Extraction Complete! ===\n";
    cout << "AIG features dir: " << aig_dir.string() << "\n";
    cout << "MIG features dir: " << mig_dir.string() << "\n";
    cout << "Statistics file: " << stats_file << "\n";
    
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////END OF FILE//////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////