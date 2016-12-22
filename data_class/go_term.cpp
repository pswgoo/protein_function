#include "go_term.h"

#include <iostream>
#include <cstdlib>
#include <exception>
#include <string>
#include <sstream>
#include <fstream>
#include <queue>
#include <map>
#include <unordered_set>
#include <math.h>
#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

using namespace std;
using namespace boost;

size_t GOTermSet::ParseGo(const std::string& go_file) {
	const int kGOIdLength = 7;
	ifstream fin(go_file);
	int node = 0;
	string line;
	while (true) {
		while (getline(fin, line))
			if (line == "[Term]")
				break;
		if (line.empty())
			break;
		GOTerm term;
		vector<int> fathers;
		vector<int> alt_ids;
		bool is_obsolete = false;
		while (getline(fin, line)) {
			if (line.empty())
				break;
			if (starts_with(line, "id: GO:"))
				term.set_id(atoi(line.substr(strlen("id: GO:")).data()));
			else if (starts_with(line, "alt_id: GO:"))
				alt_ids.push_back(atoi(line.substr(strlen("alt_id: GO:")).data()));
			else if (starts_with(line, "name: "))
				term.set_name(line.substr(strlen("name: ")).data());
			else if (starts_with(line, "namespace: "))
				term.set_type(line.substr(strlen("namespace: ")).data());
			else if (starts_with(line, "is_a: GO:"))
				fathers.push_back(atoi(line.substr(strlen("is_a: GO:"), 7).data()));
// 			else if (starts_with(line, "relationship: ")) {
// 				size_t fid = line.find("GO:");
// 				if (fid != line.npos)
// 					fathers.push_back(atoi(line.substr(fid + 3, 7).data()));
// 			}
 			else if (starts_with(line, "relationship: part_of GO:"))
 				fathers.push_back(atoi(line.substr(strlen("relationship: part_of GO:"), 7).data()));
			else if (starts_with(line, "is_obsolete: true"))
				is_obsolete = true;
		}
		if (!is_obsolete && term.id()) {
			sort(alt_ids.begin(), alt_ids.end());
			term.set_alt_ids(alt_ids);
			term.set_fathers(fathers);
			add_go_term(term);
		}
	}
	clog << "Total load " << go_terms().size() << " go terms " << endl;
	return go_terms().size();
}

void GOTermSet::Save(const std::string& file_name) const {
	ofstream fout(file_name);
	boost::archive::binary_oarchive oa(fout);
	oa << *this;
	fout.close();
}

size_t GOTermSet::Load(const std::string& file_name) {
	go_terms_.clear();

	ifstream fin(file_name);
	boost::archive::binary_iarchive ia(fin);
	ia >> *this;
	clog << "Total load " << go_terms_.size() << " go terms!" << endl;
	fin.close();
	return go_terms_.size();
}

std::vector<int> GOTermSet::FindAncestors(const std::vector<int>& go_term_ids) const {
	unordered_set<int> ancestors;
	queue<int> que;
	for (int g : go_term_ids) {
		if (ancestors.count(g) == 0)
			que.push(g);
		ancestors.insert(g);
	}
	while(!que.empty()) {
		int top = que.front();
		que.pop();
		if (HasKey(top)) {
			for (int f : QueryGOTerm(top).fathers())
				if (ancestors.count(f) == 0) {
					que.push(f);
					ancestors.insert(f);
				}
		}
	}
	vector<int> ret_ancestor;
	for (int a : ancestors)
		ret_ancestor.push_back(a);
//	clog << "gold ancestors.size = " << ancestors.size() << endl;
	return ret_ancestor;
}
