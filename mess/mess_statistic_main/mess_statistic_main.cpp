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

int main() {
	//const string kWorkDir = "D:/workspace/cafa/work/";
	const string kWorkDir = "C:/psw/cafa/CAFA3/work/"; // C:/psw/cafa/CAFA3/work/
	const string kGoTermSetFile = kWorkDir + "go_160601.gotermset";
	const string kBlastPredictFile = kWorkDir + "group1_test_blast_iter3.txt";
	const string kTrainProteinSetFile = kWorkDir + "cafa3_train_161222.proteinset";
	const string kTestProteinSetFile = kWorkDir + "cafa3_test_161222.proteinset";

	GOTermSet go_set;
	go_set.Load(kGoTermSetFile);

	//StatisticGoSet(go_set);
	//return 0;

	ProteinSet train_set;
	train_set.Load(kTrainProteinSetFile);

	AnotatedStatistic(go_set, train_set);

	return 0;
	ProteinSet test_set;
	test_set.Load(kTestProteinSetFile);


	system("pause");
	clog << "Complete" << endl;
	return 0;
}