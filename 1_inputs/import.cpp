//////////////////////////////////////////////////////
//                      INPUTS                      //
// Read AIG files into AIG and MIG data structures  //
// Output to text files in: map_trans directory     //
//////////////////////////////////////////////////////

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
#include <mockturtle/io/aiger_reader.hpp>   //Include the AIG reader from lorina
#include <mockturtle/networks/mig.hpp>      //Include the MIG network library
#include <mockturtle/networks/aig.hpp>      //Include the AIG network library
// Use normal namespaces in C++
using std::cout;
using std::endl;
using std::string;
//Use mockturtle namespaces for AIG and MIG
using mockturtle::mig_network;              //Use the MIG namespace
using mockturtle::aig_network;              //Use the AIG namespace
using lorina::read_aiger;                   //Use the read_aiger function from lorina
using lorina::return_code;
using mockturtle::aiger_reader;
namespace fs = std::filesystem;             //Use the filesystem namespace

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////FUNCTIONS///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

//1. Export AIG network to text file
void export_aig_to_text(const aig_network& aig, const string& output_filename) {
    std::ofstream outfile(output_filename);
    
    if (!outfile.is_open()) {
        cout << "Error: could not open " << output_filename << " for writing\n";
        return;
    }
    
    // Write header
    outfile << "# AIG Network Export\n";
    outfile << "# Num PIs: " << aig.num_pis() << "\n";
    outfile << "# Num POs: " << aig.num_pos() << "\n";
    outfile << "# Num Gates: " << aig.num_gates() << "\n\n";
    
    // Write node information
    outfile << "# Node_ID Gate_Type Fanin_Count\n";
    aig.foreach_node([&](auto node) {
        uint32_t node_id = aig.node_to_index(node);
        int fanin_count = aig.fanin_size(node);
        string gate_type = aig.is_constant(node) ? "CONST" : 
                          aig.is_pi(node) ? "PI" : "AND";
        outfile << node_id << " " << gate_type << " " << fanin_count << "\n";
    });
    
    outfile.close();
    cout << "Exported AIG to " << output_filename << "\n";
}

//2. Export MIG network to text file
void export_mig_to_text(const mig_network& mig, const string& output_filename) {
    std::ofstream outfile(output_filename);
    
    if (!outfile.is_open()) {
        cout << "Error: could not open " << output_filename << " for writing\n";
        return;
    }
    
    // Write header
    outfile << "# MIG Network Export\n";
    outfile << "# Num PIs: " << mig.num_pis() << "\n";
    outfile << "# Num POs: " << mig.num_pos() << "\n";
    outfile << "# Num Gates: " << mig.num_gates() << "\n\n";
    
    // Write node information
    outfile << "# Node_ID Gate_Type Fanin_Count\n";
    mig.foreach_node([&](auto node) {
        uint32_t node_id = mig.node_to_index(node);
        int fanin_count = mig.fanin_size(node);
        string gate_type = mig.is_constant(node) ? "CONST" : 
                          mig.is_pi(node) ? "PI" : "MAJ";
        outfile << node_id << " " << gate_type << " " << fanin_count << "\n";
    });
    
    outfile.close();
    cout << "Exported MIG to " << output_filename << "\n";
}

/////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////MAIN///////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    if (argc != 2) {
        cout << "Error: missing file name\n"
             << "Usage: " << argv[0] << " <filename.aig>\n";
        return 1;
    }

    string filename = argv[1];

    // 3. Command the OS to reserve RAM for BOTH structures
    aig_network aig;
    mig_network mig;

    // 4. Read the file from the hard drive into the AIG RAM block
    return_code result_aig = read_aiger(filename, aiger_reader(aig));

    if (result_aig != return_code::success) {
        cout << "Error reading " << filename << " into AIG RAM!\n";
        return 1;
    } else {
        cout << "Successfully read " << filename << " into AIG RAM!\n"
             << "Number of AIG AND gates: " << aig.num_gates() << endl;
    }
    
    // 5. Read the file from the hard drive and translate it into the MIG RAM block
    return_code result_mig = read_aiger(filename, aiger_reader(mig));

    if (result_mig != return_code::success) {
        cout << "Error reading " << filename << " into MIG RAM!\n";
        return 1;
    } else {
        cout << "Successfully read " << filename << " into MIG RAM!\n"
             << "Number of MIG Majority gates: " << mig.num_gates() << endl;
    }
    
    // 6. Create output directory if it doesn't exist
    string map_trans_dir = "map_trans";
    if (!fs::exists(map_trans_dir)) {
        fs::create_directory(map_trans_dir);
        cout << "Created directory: " << map_trans_dir << "\n";
    }
    
    // 7. Export network structures to map_trans
    string basename = fs::path(filename).stem();  // Get filename without extension
    string aig_net_output = map_trans_dir + "/" + basename + "_aig.txt";
    string mig_net_output = map_trans_dir + "/" + basename + "_mig.txt";
    
    export_aig_to_text(aig, aig_net_output);
    export_mig_to_text(mig, mig_net_output);
    
    cout << "\n=== Stage 1 Complete ===\n";
    cout << "Networks loaded to RAM successfully!\n";
    cout << "Next: Run extract_features in 2_process to extract matrices\n";

    return 0;

}
///////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////END OF FILE////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////