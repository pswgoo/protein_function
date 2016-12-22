#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <queue>
#include <unordered_map>

#include <boost/algorithm/string.hpp>

#include "data_class/protein_sequence.h"
#include "data_class/go_term.h"
#include "learning/evaluation.h"

using namespace std;
using namespace boost;
std::unordered_set<std::string> kExp = { "EXP","IDA","IPI","IMP","IGI","IEP","TAS","IC" };
const vector<string> kLeftSpecies = { "BACSU","PSEAE","STRPN","ARATH","YEAST","XENLA","HUMAN","MOUSE","RAT","DICDI","ECOLI" };

struct BlastPredictInstance {
	string protein_;
	vector<string> similar_proteins_;
	vector<double> similar_evalues_;
	vector<double> similar_bitscore_;
};

vector<BlastPredictInstance> ParseBlastPredictResult(const std::string& filename, const ProteinSet& test_set) {
	const size_t kEvalueIndex = 10;
	const size_t kBitscoreIndex = 11;

	vector<BlastPredictInstance> ret_instances;
	ifstream fin(filename);
	string last_line;
	string line;
	while (true) {
		if (last_line.empty()) {
			while (getline(fin, line)) {
				if (line.front() != '#')
					break;
			}
		}
		else
			line = last_line;

		if (line.empty())
			break;
		BlastPredictInstance instance;
		vector<string> tokens;
		split(tokens, line, is_any_of("\t"));
		instance.protein_ = tokens[0];
		instance.similar_proteins_.push_back(tokens[1]);
		instance.similar_evalues_.push_back(atof(tokens[kEvalueIndex].data()));
		instance.similar_bitscore_.push_back(atof(tokens[kBitscoreIndex].data()));
		last_line.clear();
		while (getline(fin, line)) {
			if (line.front() == '#')
				break;
			tokens.clear();
			split(tokens, line, is_any_of("\t"));
			string name = tokens[0].substr(tokens[0].find_last_of('|') + 1);
			if (name != instance.protein_) {
				last_line = line;
				break;
			}
			instance.similar_proteins_.push_back(tokens[1].substr(tokens[1].find_last_of('|') + 1));
			instance.similar_evalues_.push_back(atof(tokens[kEvalueIndex].data()));
			instance.similar_bitscore_.push_back(atof(tokens[kBitscoreIndex].data()));
		}
		ret_instances.push_back(instance);
	}
	return ret_instances;
}

MultiLabelPredictAnswer BlastPredict(const ProteinSet& train_set, const GOTermSet& goterm_set, const BlastPredictInstance& test_instance, GoType go_type) {
	unordered_map<int, double> label_score;
	for (size_t i = 0; i < test_instance.similar_proteins_.size(); ++i) {
		const Protein &protein = train_set[test_instance.similar_proteins_[i]];
		double score = test_instance.similar_evalues_[i];
		vector<int> go_leaves;
		for (const int go : protein.go_term(go_type)) {
			if (goterm_set.HasKey(go))
				go_leaves.push_back(go);
			else
				cerr << "Error: go: " << go << " cannot find" << endl;
		}
		for (int go_id : goterm_set.FindAncestors(go_leaves)) {
			if (label_score.count(go_id) == 0)
				label_score[go_id] = score;
			else
				label_score[go_id] = max(label_score[go_id], score);
		}
	}
	MultiLabelPredictAnswer answer;
	for (auto pr : label_score)
		answer.push_back(pr);
	sort(answer.begin(), answer.end(), [](const pair<int, double>& p1, const pair<int, double>& p2) {return p1.second > p2.second; });

	return answer;
}

vector<MultiLabelPredictAnswer> BlastPredict(const ProteinSet& train_set, const GOTermSet& goterm_set, const vector<BlastPredictInstance>& test_instances, GoType go_type) {
	vector<MultiLabelPredictAnswer> predict_answers(test_instances.size());
#pragma omp parallel for schedule(dynamic)
	for (int64_t i = 0; i < test_instances.size(); ++i)
		predict_answers[i] = BlastPredict(train_set, goterm_set, test_instances[i], go_type);
	
	return predict_answers;
}

int main() {

	return 0;
}