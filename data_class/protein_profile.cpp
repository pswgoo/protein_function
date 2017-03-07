#include "protein_profile.h"

#include <iostream>
#include <cstdlib>
#include <exception>
#include <string>
#include <sstream>

#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

using namespace std;
using namespace boost;

static const boost::property_tree::ptree& EmptyPtree() {
	static boost::property_tree::ptree t;
	return t;
}

std::string ProteinProfile::ToString() const {
	string ret_value("Protein profile: \n");

	ret_value += "name: " + name() + "\n";
	ret_value += "species: " + species() + "\n";
	ret_value += "accessions:";
	for (auto &u : accessions())
		ret_value += " " + u;
	ret_value += "\n";
	ret_value += "ref pmids: " + to_string(ref_pmids().size()) + "\n";
	for (auto u : ref_pmids())
		ret_value += " " + to_string(u);
	ret_value += "\n";
	return ret_value;
}

size_t ProteinProfile::ParseUniprotPtree(const boost::property_tree::ptree& rt_tree) {
	using namespace boost::property_tree;

	if (rt_tree.empty())
		return 0;

	BOOST_FOREACH(const ptree::value_type& v, rt_tree) {
		if (v.first == "accession") {
			add_accession(v.second.data());
		}
		else if (v.first == "name") {
			vector<string> tokens;
			boost::split(tokens, v.second.data(), is_any_of("_"));
			if (tokens.size() != 2)
				cerr << "Warning: name tokens not equal to 2" << endl;
			else {
				set_name(tokens[0]);
				set_species(tokens[1]);
			}
		}
		else if (v.first == "reference") {
			const ptree &ptree_citation = v.second.get_child("citation", EmptyPtree());
			BOOST_FOREACH(const ptree::value_type& v2, ptree_citation) {
				if (v2.first == "dbReference") {
					string db = boost::to_upper_copy(v2.second.get("<xmlattr>.type", string()));
					if (db == "PUBMED") {
						int pmid = v2.second.get("<xmlattr>.id", -1);
						if (pmid < 0)
							cerr << "Warning: pmid not valid" << endl;
						else
							add_ref_pmid(pmid);
					}
				}
			}
		}
	}
	return 1;
}

size_t ProteinProfile::ParseUniprotXml(const std::string& xml_file) {
	using namespace boost::property_tree;

	ptree rt_tree;
	try {
		read_xml(xml_file, rt_tree);
	}
	catch (const std::exception& e) {
		cerr << "Error: " << e.what() << endl;
		return 0;
	}

	size_t success_cnt = 0;
	const ptree ptree_entry = rt_tree.get_child("uniprot.entry", EmptyPtree());
	if (ParseUniprotPtree(ptree_entry) > 0)
		++success_cnt;
	return success_cnt;
}

size_t ProteinProfileSet::ParseUniprotXml(const std::string& xml_file) {
	using namespace boost::property_tree;

	const string kXmlHead = "<entry ";
	const string kXmlTail = "</entry>";

	protein_profiles_.clear();
	protein_indices_.clear();

	ifstream fin(xml_file);
	string line;
	string xml_str;
	bool b_begin = false;
	size_t success_cnt = 0;
	while (getline(fin, line)) {
		string tmp_str = trim_left_copy(line);
		if (strncmp(tmp_str.c_str(), kXmlHead.c_str(), kXmlHead.size()) == 0)
			b_begin = true;
		if (b_begin)
			xml_str += line + "\n";
		if (strncmp(tmp_str.c_str(), kXmlTail.c_str(), kXmlTail.size()) == 0) {
			istringstream sin(xml_str);

			ptree rt_tree;
			try {
				read_xml(sin, rt_tree);
			}
			catch (const std::exception& e) {
				cerr << "Error: " << e.what() << endl;
			}
			ProteinProfile protein_profile;
			size_t tmp_ret = protein_profile.ParseUniprotPtree(rt_tree.get_child("entry", EmptyPtree()));
			if (tmp_ret > 0) {
				protein_profiles_.push_back(protein_profile);
				success_cnt += tmp_ret;
			}
			else
				cerr << "Warning: Parse ptree error!, name = " << protein_profile.name() << endl;

			if (success_cnt % 10000 == 0)
				clog << "\rLoaded " << success_cnt << " successfully";

			xml_str.clear();
			b_begin = false;
		}
	}
	clog << endl;
	fin.close();

	if (success_cnt != protein_profiles_.size())
		cerr << "Error: success_cnt != protein_profiles_.size()" << endl;

	for (int i = 0; i < protein_profiles_.size(); ++i)
		for (int j = 0; j < protein_profiles_[i].accessions().size(); ++j)
			protein_indices_[protein_profiles_[i].accessions()[j]] = i;

	clog << "Total loaded " << success_cnt << " instances successfully" << endl;
	return success_cnt;
}

void ProteinProfileSet::Save(const std::string& file_name) const {
	ofstream fout(file_name, ios::binary);
	boost::archive::binary_oarchive oa(fout);
	oa << *this;
	fout.close();
}

size_t ProteinProfileSet::Load(const std::string& file_name) {
	protein_profiles_.clear();
	protein_indices_.clear();

	ifstream fin(file_name, ios::binary);
	boost::archive::binary_iarchive ia(fin);
	ia >> *this;
	fin.close();
	clog << "Total load " << protein_profiles_.size() << " protein sequences!" << endl;
	return protein_profiles_.size();
}
