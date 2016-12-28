#include "protein_sequence.h"

#include <iostream>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>
#include <sstream>
#include <cctype>

#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

using namespace std;
using namespace boost;

AminoType GetAminoType(const std::string &x) {
	for	(int name = AminoType::NON; name <= AminoType::VAL; ++name)
		if (kTypeString[name] == x || (x.length() == 1 && x[0] == kTypeChar[name]))
			return static_cast<AminoType>(name);
	return AminoType::NON;
}

void ProteinSet::Save(const std::string& file_name) const {
	ofstream fout(file_name, ios_base::binary);
	boost::archive::binary_oarchive oa(fout);
	oa << *this;
	fout.close();
}

int ProteinSet::Load(const std::string& file_name) {
	protein_indices_.clear();
	proteins_.clear();

	ifstream fin(file_name, ios_base::binary);
	boost::archive::binary_iarchive ia(fin);
	ia >> *this;
	clog << "Total load " << proteins_.size() << " proteins!" << endl;
	fin.close();
	return (int)proteins_.size();
}

std::string RemoveSpace(const std::string& str) {
	std::string ret = str;
	ret.resize(remove_if(ret.begin(), ret.end(), isspace) - ret.begin());
	return ret;
}

int ProteinSet::ParseRawTxt(const std::string & sequence_file, const std::string & mf_go_file, const std::string & bp_go_file, const std::string & cc_go_file, bool only_left_indexed) {
	ifstream fin(sequence_file);

	string line;
	while (fin.peek() == '>') {
		getline(fin, line);
		string id = line.substr(1);
		string seq;
		while (fin.peek() != '>' && fin.peek() != std::char_traits<char>::eof()) {
			getline(fin, line);
			seq += line;
		}
		seq = RemoveSpace(seq);
		protein_indices_[id] = (int)proteins_.size();
		proteins_.push_back(Protein(id, seq));
	}

	auto load_go = [&](string file_name, GoType go_type){
		ifstream in(file_name);
		string p_id, go;
		while (in >> p_id >> go) {
			if (protein_indices_.count(p_id) > 0) {
				go = go.substr(1 + go.find(":"));
				proteins_[protein_indices_.at(p_id)].go_terms_[go_type].push_back(stoi(go));
			}
		}
	};

	load_go(mf_go_file, MF);
	load_go(bp_go_file, BP);
	load_go(cc_go_file, CC);

	int parsed_cnt = (int)proteins_.size();
	vector<Protein> indexed_proteins;
	for (int i = 0; i < proteins_.size(); ++i)
		if (!proteins_[i].go_terms_[MF].empty() || !proteins_[i].go_terms_[BP].empty() || !proteins_[i].go_terms_[CC].empty())
			indexed_proteins.push_back(proteins_[i]);
	if (only_left_indexed)
		set_proteins(indexed_proteins);
	clog << "Total parsed " << parsed_cnt << " proteins, " << indexed_proteins.size() << " have go terms" << endl;
	return (int)proteins_.size();
}
