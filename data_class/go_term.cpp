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

std::vector<int> GOTermSet::ToAnnotationVector(std::vector<int>& go_term_ids) const {
	vector<int> ret(go_terms_.size(), 0);
	for (int go : go_term_ids)
		if (go_term_index_.count(go) > 0)
			ret[go_term_index_.at(go)] = 1;
		else
			cerr << "GO: " << go << " not found" << endl;
	return ret;
}

std::vector<int> GOTermSet::ToGoIds(std::vector<int>& go_annotation_vector) const {
	vector<int> ret;
	for (int i = 0; i < go_annotation_vector.size(); ++i)
		if (go_annotation_vector[i] > 0)
			ret.push_back(go_terms_[i].id());
	return ret;
}

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
	ofstream fout(file_name, ios_base::binary);
	boost::archive::binary_oarchive oa(fout);
	oa << *this;
	fout.close();
}

size_t GOTermSet::Load(const std::string& file_name) {
	go_terms_.clear();
	go_term_index_.clear();

	ifstream fin(file_name, ios_base::binary);
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
		if (ancestors.count(g) == 0 && HasKey(g) && !QueryGOTerm(g).fathers().empty()) {
			que.push(g);
			ancestors.insert(g);
		}
	}
	while(!que.empty()) {
		int top = que.front();
		que.pop();
		if (HasKey(top)) {
			for (int f : QueryGOTerm(top).fathers())
				if (ancestors.count(f) == 0 && HasKey(f) && !QueryGOTerm(f).fathers().empty()) {
					que.push(f);
					ancestors.insert(f);
				}
		}
		else
			clog << "Warning: " << top << " cannot find in go_set" << endl;
	}
	vector<int> ret_ancestor;
	for (int a : ancestors)
		ret_ancestor.push_back(a);
//	clog << "gold ancestors.size = " << ancestors.size() << endl;
	return ret_ancestor;
}

vector<pair<int, double>> GOTermSet::ScoreAncestors(const std::vector<float>& vec_scores, float cutoff) const {
	unordered_map<int, double> scores;
	unordered_set<int> visited;
	queue<int> que;
	vector<bool> is_leaf(go_terms().size(), true);
	for (int i = 0; i < go_terms().size(); ++i)
		for (int f : go_terms()[i].fathers())
			is_leaf[go_term_index_.at(f)] = false;
	for (int i = 0; i < is_leaf.size(); ++i) {
		scores[go_terms()[i].id()] = vec_scores[i];
		if (is_leaf[i]) {
			que.push(go_terms()[i].id());
			visited.insert(go_terms()[i].id());
		}
	}

	while (!que.empty()) {
		int top = que.front();
		que.pop();
		if (HasKey(top)) {
			double top_score = scores[top];
			for (int f : QueryGOTerm(top).fathers())
				if (HasKey(f) && !QueryGOTerm(f).fathers().empty() && (visited.count(f) == 0 || scores[f] < top_score)) {
					scores[f] = max(top_score, scores[f]);
					que.push(f);
					visited.insert(f);
				}
		}
	}

	vector<pair<int, double>> ret;
	for (const pair<int, double>& pr : scores)
		if (pr.second > cutoff)
			ret.push_back(pr);

	return ret;
}
