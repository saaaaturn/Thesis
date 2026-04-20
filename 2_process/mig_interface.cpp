/* HybridSYN native MIG executor.
 *
 * Reads an input AIGER circuit, loads it into an MIG network, applies a native
 * MIG optimization selected by name, cleans the result, and writes the updated
 * circuit back out as AIGER.
 */

#include <filesystem>
#include <functional>
#include <iostream>
#include <string>

#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/mig_algebraic_rewriting.hpp>
#include <mockturtle/algorithms/node_resynthesis.hpp>
#include <mockturtle/algorithms/node_resynthesis/mig_npn.hpp>
#include <mockturtle/algorithms/rewrite.hpp>
#include <mockturtle/io/aiger_reader.hpp>
#include <mockturtle/io/write_aiger.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/networks/mig.hpp>
#include <mockturtle/utils/tech_library.hpp>
#include <mockturtle/views/depth_view.hpp>

namespace fs = std::filesystem;

using mockturtle::aig_network;
using mockturtle::cleanup_dangling;
using mockturtle::exact_library;
using mockturtle::mig_algebraic_depth_rewriting;
using mockturtle::mig_network;
using mockturtle::mig_npn_resynthesis;
using mockturtle::node_resynthesis;
using mockturtle::node_resynthesis_params;
using mockturtle::rewrite;
using mockturtle::rewrite_params;
using mockturtle::write_aiger;

namespace
{

struct options
{
	std::string action;
	std::string input;
	std::string output;
};

void print_usage()
{
	std::cerr << "Usage: interface --action <mig_action> --input <input.aig> --output <output.aig>\n";
	std::cerr << "   or: interface <mig_action> <input.aig> <output.aig>\n";
}

bool parse_args( int argc, char** argv, options& opt )
{
	if ( argc == 4 )
	{
		opt.action = argv[1];
		opt.input = argv[2];
		opt.output = argv[3];
		return true;
	}

	if ( argc != 7 )
	{
		return false;
	}

	for ( int i = 1; i < argc; i += 2 )
	{
		const std::string key = argv[i];
		const std::string value = argv[i + 1];

		if ( key == "--action" )
		{
			opt.action = value;
		}
		else if ( key == "--input" )
		{
			opt.input = value;
		}
		else if ( key == "--output" )
		{
			opt.output = value;
		}
		else
		{
			return false;
		}
	}

	return !opt.action.empty() && !opt.input.empty() && !opt.output.empty();
}

template<typename ActionFn>
int run_native_mig_action( const options& opt, ActionFn&& action_fn )
{
	mig_network mig;

	if ( lorina::read_aiger( opt.input, mockturtle::aiger_reader( mig ) ) != lorina::return_code::success )
	{
		std::cerr << "Failed to read input AIGER file: " << opt.input << '\n';
		return 1;
	}

	action_fn( mig );
	mig = cleanup_dangling( mig );

	const auto aig = cleanup_dangling<mig_network, aig_network>( mig );
	const fs::path output_path = opt.output;
	if ( !output_path.parent_path().empty() )
	{
		std::error_code ec;
		fs::create_directories( output_path.parent_path(), ec );
		if ( ec )
		{
			std::cerr << "Failed to create output directory: " << output_path.parent_path() << '\n';
			return 1;
		}
	}

	write_aiger( aig, opt.output );
	return 0;
}

} // namespace

int main( int argc, char** argv )
{
	options opt;
	if ( !parse_args( argc, argv, opt ) )
	{
		print_usage();
		return 1;
	}

	if ( opt.action == "mig_balance" )
	{
		return run_native_mig_action( opt, []( mig_network& mig ) {
			mockturtle::depth_view depth_mig{ mig };
			mig_algebraic_depth_rewriting( depth_mig );
		} );
	}

	if ( opt.action == "mig_rewrite" )
	{
		return run_native_mig_action( opt, []( mig_network& mig ) {
			mig_npn_resynthesis resyn{ true };
			exact_library<mig_network> exact_lib( resyn );
			rewrite_params ps;
			rewrite( mig, exact_lib, ps );
		} );
	}

	if ( opt.action == "mig_refactor" )
	{
		return run_native_mig_action( opt, []( mig_network& mig ) {
			mig_network refined;
			mig_npn_resynthesis resyn{ false };
			node_resynthesis_params ps;
			node_resynthesis( refined, mig, resyn, ps );
			mig = std::move( refined );
		} );
	}

	if ( opt.action == "mig_resub" )
	{
		return run_native_mig_action( opt, []( mig_network& mig ) {
			mig_network refined;
			mig_npn_resynthesis resyn{ true };
			node_resynthesis_params ps;
			node_resynthesis( refined, mig, resyn, ps );
			mig = std::move( refined );
		} );
	}

	if ( opt.action == "mig_rewrite_z" )
	{
		return run_native_mig_action( opt, []( mig_network& mig ) {
			mig_npn_resynthesis resyn{ true };
			exact_library<mig_network> exact_lib( resyn );
			rewrite_params ps;
			ps.allow_zero_gain = true;
			rewrite( mig, exact_lib, ps );
		} );
	}

	if ( opt.action == "mig_cleanup" )
	{
		return run_native_mig_action( opt, []( mig_network& mig ) {
			mig = cleanup_dangling( mig );
		} );
	}

	std::cerr << "Unsupported MIG action: " << opt.action << '\n';
	print_usage();
	return 1;
}
