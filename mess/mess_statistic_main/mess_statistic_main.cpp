#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <set>
#include <map>
#include <queue>
#include <memory>
#include <unordered_map>

#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "data_class/protein_profile.h"
#include "data_class/protein_sequence.h"
#include "data_class/go_term.h"
#include "learning/evaluation.h"

using namespace std;
using namespace boost;

void StatisticGoSet(const GOTermSet& go_set) {
	map<int, int> go_depth_cnt[GO_TYPE_SIZE];
	for (int i = 0; i < go_set.go_terms().size(); ++i) {
		int id = go_set.go_terms()[i].id();
		queue<pair<int, int>> que;
		que.push({ id, 0 });
		set<int> visited;
		visited.insert(id);
		int depth = 0;
		while (!que.empty()) {
			pair<int,int> top = que.front();
			que.pop();
			if (go_set.QueryGOTerm(top.first).fathers().empty()) {
				depth = top.second;
				break;
			}
			for (int f : go_set.QueryGOTerm(top.first).fathers())
				if (visited.count(f) == 0)
					que.push({ f, top.second + 1 });
		}
		GoType type = GoTypeStrToTypeId(go_set.go_terms()[i].type());
		go_depth_cnt[type][depth]++;
	}
	for (int i = 0; i < GO_TYPE_SIZE; ++i) {
		ofstream fout(kGoTypeStr[i] + "_count.csv");
		fout << "depth,go_count" << endl;
		for (pair<int, int> pr : go_depth_cnt[i])
			fout << pr.first << "," << pr.second << endl;
	}
}

void AnotatedStatistic(const GOTermSet& go_set, const ProteinSet & train_set) {
	map<int, int> go_indexing_cnt[GO_TYPE_SIZE];
	for (int i = 0; i < train_set.Size(); ++i) {
		for (int go_type = 0; go_type < GO_TYPE_SIZE; ++go_type) {
			vector<int> indexed_gos = go_set.FindAncestors(train_set[i].go_term(GoType(go_type)));
			for (int g : indexed_gos)
				go_indexing_cnt[go_type][g]++;
		}
	}

	ofstream fout("go_train_instance_count.csv");
	fout << "go_id,go_type,instance_count" << endl;
	for (int go_type = 0; go_type < GO_TYPE_SIZE; ++go_type)
		for (pair<int, int> pr : go_indexing_cnt[go_type])
			fout << pr.first << "," << kGoTypeStr[go_type] << "," << pr.second << endl;
}

void PmidStatistic(const ProteinProfileSet& profile_set, const ProteinSet& train_set) {
	int indexed_cnt = 0;
	int has_pmid_cnt = 0;
	int sum_pmid = 0;
	int cannot_cnt = 0;
	for (int i = 0; i < train_set.Size(); ++i)
		if (train_set[i].Indexed()) {
			if (profile_set.Has(train_set[i].id_)) {
				int pmid_num = profile_set.Query(train_set[i].id_).ref_pmids().size();
				if (pmid_num > 0)
					++has_pmid_cnt;
				sum_pmid += pmid_num;
			}
			else {
				cannot_cnt++;
				cerr << train_set[i].id_ << " not found!" << endl;
			}
			++indexed_cnt;
		}
	clog << "Cannot find cnt " << cannot_cnt << endl;
	clog << "Train set size: " << train_set.Size() << endl;
	clog << "Train set indexed size: " << indexed_cnt << endl;
	clog << "Train set has_pmid_cnt: " << has_pmid_cnt << endl;
	clog << "Train set average pmids: " << sum_pmid / double(has_pmid_cnt) << endl;
}

void CountSequenceLengthAnotationNum(const GOTermSet& go_set, const ProteinSet & train_set) {
	ofstream fout("sequence_annation.csv");
	fout << "id,sequence length,mf count,bp count,cc count" << endl;
	for (int i = 0; i < train_set.Size(); ++i)
		fout << train_set[i].id_ <<"," << train_set[i].sequence_.size() << "," 
		<< go_set.FindAncestors(train_set[i].go_term(MF)).size() << ","
		<< go_set.FindAncestors(train_set[i].go_term(BP)).size() << ","
		<< go_set.FindAncestors(train_set[i].go_term(CC)).size() << "," << endl;
}

int main() {
	const string kWorkDir = "C:/psw/cafa/protein_cafa2/work/";
	//const string kWorkDir = "C:/psw/cafa/CAFA3/work/"; // C:/psw/cafa/CAFA3/work/
	const string kGoTermSetFile = kWorkDir + "go_160601.gotermset";
	const string kBlastPredictFile = kWorkDir + "group1_test_blast_iter3.txt";
	const string kTrainProteinSetFile = kWorkDir + "cafa3_train_161222.proteinset";
	const string kTestProteinSetFile = kWorkDir + "cafa3_test_161222.proteinset";
	const string kProteinProfileSetFile = kWorkDir + "uniprot_sprot_201610.profileset";

	GOTermSet go_set;
	go_set.Load(kWorkDir + "go_140101.gotermset");

	ProteinSet train_set;
	train_set.Load(kWorkDir + "cafa2_train_170307.proteinset");

	int mn_id = 1000000, mx_id = -1;
	for (int i = 0; i < go_set.go_terms().size(); ++i) {
		mn_id = min(mn_id, go_set.go_terms()[i].id());
		mx_id = max(mx_id, go_set.go_terms()[i].id());
	}
	clog << "GoSet.size: " << go_set.go_terms().size() << endl;
	clog << "GoSet.min id: " << mn_id << endl;
	clog << "GoSet.max id: " << mx_id << endl;
	AnotatedStatistic(go_set, train_set);
	return 0;

	//StatisticGoSet(go_set);
	//return 0;

	CountSequenceLengthAnotationNum(go_set, train_set);
	return 0;

	ProteinProfileSet profile_set;
	profile_set.Load(kProteinProfileSetFile);
	PmidStatistic(profile_set, train_set);
	//AnotatedStatistic(go_set, train_set);
	
	system("pause");
	return 0;
	ProteinSet test_set;
	test_set.Load(kTestProteinSetFile);


	system("pause");
	clog << "Complete" << endl;
	return 0;
}