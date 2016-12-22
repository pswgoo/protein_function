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
std::unordered_set<std::string> kExp = {"EXP","IDA","IPI","IMP","IGI","IEP","TAS","IC"};
const vector<string> kLeftSpecies = {"BACSU","PSEAE","STRPN","ARATH","YEAST","XENLA","HUMAN","MOUSE","RAT","DICDI","ECOLI"};

const string kBP = "biological_process";
const string kMF = "molecular_function";

struct BlastPredictInstance {
	string protein_;
	vector<string> similar_proteins_;
	vector<double> similar_evalues_;
	vector<double> similar_identities_;
};

vector<BlastPredictInstance> ParseBlastPredictResult(const std::string& filename) {
	const size_t kEvalueIndex = 10;
	const size_t kIdentityIndex = 2;

	vector<BlastPredictInstance> ret_instances;
	ifstream fin(filename);
	string last_line;
	string line;
	while (true) {
		if (last_line.empty()) {
			while(getline(fin, line)) {
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
		instance.protein_ = tokens[0].substr(tokens[0].find_last_of('|') + 1);
		instance.similar_proteins_.push_back(tokens[1].substr(tokens[1].find_last_of('|') + 1));
		instance.similar_evalues_.push_back(atof(tokens[kEvalueIndex].data()));
		instance.similar_identities_.push_back(atof(tokens[kIdentityIndex].data()));
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
			instance.similar_identities_.push_back(atof(tokens[kIdentityIndex].data()));
		}
		ret_instances.push_back(instance);
	}
	return ret_instances;	
}

void OutputInstance(string filename, const vector<BlastPredictInstance>& instances) {
	ofstream fout(filename);
	for (const auto& instance : instances) {
		for (size_t i = 0; i < instance.similar_proteins_.size(); ++i)
			fout << instance.protein_ << " " << instance.similar_proteins_[i] << " " << instance.similar_identities_[i]  << " " << instance.similar_evalues_[i] << endl;
		fout << endl;
	}
}

MultiLabelPredictAnswer BlastPredict(const ProteinSequenceSet& train_set, const GOTermSet& goterm_set, const BlastPredictInstance& test_instance) {
	unordered_map<int, double> label_score;
	for (size_t i = 0; i < test_instance.similar_proteins_.size(); ++i) {
		const ProteinSequence &protein = train_set.QueryByName(test_instance.similar_proteins_[i]);
		double score = test_instance.similar_identities_[i];
		vector<int> go_leaves;
		for (const ProteinSequence::GOType& go : protein.go_terms()) {
			if (goterm_set.HasKey(go.id_))
				go_leaves.push_back(goterm_set.QueryGOTerm(go.id_).id());
			else
				cerr << "Error: go: " << go.id_ << " cannot find" << endl;
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
	sort(answer.begin(), answer.end(), [](const pair<int,double>& p1, const pair<int,double>& p2) {return p1.second > p2.second;});
	
	return answer;
}

vector<MultiLabelPredictAnswer> BlastPredict(const ProteinSequenceSet& train_set, const GOTermSet& goterm_set, const vector<BlastPredictInstance>& test_instances) {
	vector<MultiLabelPredictAnswer> predict_answers(test_instances.size());
	#pragma omp parallel for schedule(dynamic)
	for (int64_t i = 0;i < test_instances.size(); ++i) {
		predict_answers[i] = BlastPredict(train_set, goterm_set, test_instances[i]);
	}
	return predict_answers;
}

vector<MultiLabelPredictAnswer> PriorPredict(const ProteinSequenceSet& train_set, const GOTermSet& goterm_set, const vector<BlastPredictInstance>& test_instances) {
	unordered_map<int, int> go_cnt;
	for (const ProteinSequence& protein : train_set.protein_sequences()) {
		vector<int> go_leaves;
		for (const ProteinSequence::GOType& go : protein.go_terms()) {
			if (goterm_set.HasKey(go.id_))
				go_leaves.push_back(goterm_set.QueryGOTerm(go.id_).id());
			else
				cerr << "Error: go: " << go.id_ << " cannot find in prior predict" << endl;
		}
		for (int go_id : goterm_set.FindAncestors(go_leaves)) {
			if (go_cnt.count(go_id) == 0)
				go_cnt[go_id] = 0;
			++go_cnt[go_id];
		}
	}
	vector<pair<int,double> > term_freq;
	for (auto pr : go_cnt)
		term_freq.push_back({pr.first, pr.second / double(train_set.protein_sequences().size())});
	sort(term_freq.begin(), term_freq.end(), [](pair<int, double> p1, pair<int, double> p2) { return p1.second > p2.second; });
	vector<MultiLabelPredictAnswer> ret_answers(test_instances.size(), term_freq);
	return ret_answers;
}

vector<MultiLabelPredictAnswer> PriorPredict(const ProteinSequenceSet& train_set, const GOTermSet& goterm_set, const vector<BlastPredictInstance>& test_instances, const string& predict_type) {
	unordered_map<int, int> go_cnt;
	for (const ProteinSequence& protein : train_set.protein_sequences()) {
		vector<int> go_leaves;
		for (const ProteinSequence::GOType& go : protein.go_terms()) {
			if (goterm_set.HasKey(go.id_) && goterm_set.QueryGOTerm(go.id_).type() == predict_type)
				go_leaves.push_back(goterm_set.QueryGOTerm(go.id_).id());
			else if (!goterm_set.HasKey(go.id_))
				cerr << "Error: go: " << go.id_ << " cannot find in prior predict" << endl;
		}
		for (int go_id : goterm_set.FindAncestors(go_leaves)) {
			if (goterm_set.QueryGOTerm(go_id).type() != predict_type)
				continue;
			if (go_cnt.count(go_id) == 0)
				go_cnt[go_id] = 0;
			++go_cnt[go_id];
		}
	}
	vector<pair<int,double> > term_freq;
	for (auto pr : go_cnt)
		term_freq.push_back({pr.first, pr.second / double(train_set.protein_sequences().size())});
	sort(term_freq.begin(), term_freq.end(), [](pair<int, double> p1, pair<int, double> p2) { return p1.second > p2.second; });
	vector<MultiLabelPredictAnswer> ret_answers(test_instances.size(), term_freq);
	return ret_answers;
}

void OutputPredictResult(string filename, vector<BlastPredictInstance>& blast_instances, const vector<MultiLabelGoldAnswer>& gold_standard, const vector<MultiLabelPredictAnswer>& prediction) {
	const int kMaxPredictNum = 30;
	ofstream fout(filename);
	int empty_gold_cnt = 0;
	for (size_t i = 0; i < blast_instances.size(); ++i) {
		if (gold_standard[i].empty()) {
			++empty_gold_cnt;
			continue;
		}
		fout << blast_instances[i].protein_ << " " << gold_standard[i].size();
		for (int g : gold_standard[i])
			fout << " " << g;
		fout << endl;
		fout << prediction[i].size();
		for (size_t j = 0; j < prediction[i].size(); ++j) {
			fout << " " << prediction[i][j].first << ":" << prediction[i][j].second;
		}
		fout << endl;
	}
	clog << "Empty gold instance is " << empty_gold_cnt << " / " << blast_instances.size() << endl;
}

vector<pair<int, double>> GetAncestorScore(const GOTermSet& go_set, const vector<pair<int, double>>& child_score) {
	vector<pair<int, double>> ret_ancestor;
	unordered_map<int, double> ancestors;
	queue<int> que;
	for (auto g : child_score) {
		if (ancestors.count(g.first) == 0)
			que.push(g.first);
		ancestors.insert(g);
	}
	while(!que.empty()) {
		int top = que.front();
		double score = ancestors[top];
		que.pop();
		if (go_set.HasKey(top)) {
			for (int f : go_set.QueryGOTerm(top).fathers()) {
//				if (!go_set.HasKey(f))
//					continue;
				if (ancestors.count(f) == 0) {
					que.push(f);
					ancestors.insert({f, score});
				}
				else if (score > ancestors[f])
					ancestors[f] = score;
			}
		}
	}
//	clog << "ancestors.size = " << ancestors.size() << endl;
	for (auto a : ancestors)
		ret_ancestor.push_back(a);
	return ret_ancestor;
}

void CountGOTermSet() {
	GOTermSet go_set;
	go_set.ParseGo("gene_ontology_edit.obo.2010-06-01");
	go_set.Save("go_term_set_20100601.gotermset");
	unordered_set<int> mf_go, bp_go;
	unordered_set<int> has_child_node;
	for (GOTerm term : go_set.go_terms()) {
		if (term.type() == kMF)
			mf_go.insert(term.id());
		if (term.type() == kBP)
			bp_go.insert(term.id());
		for (int f : term.fathers())
			has_child_node.insert(go_set.QueryGOTerm(f).id());
	}
	int mf_leaf_cnt = 0;
	for (int n : mf_go)
		if (has_child_node.count(n) == 0)
			++mf_leaf_cnt;
	int bp_leaf_cnt = 0;
	ofstream fout("bp_leaf.txt");
	for (int n : bp_go)
		if (has_child_node.count(n) == 0) {
			++bp_leaf_cnt;
			fout << n << endl;
		}
	cout << "mf total count: " << mf_go.size() << ", mf leaf count: " << mf_leaf_cnt << endl; 
	cout << "bp total count: " << bp_go.size() << ", bp leaf count: " << bp_leaf_cnt << endl; 
}

void CountTestSetGoDiff() {
	ProteinSequenceSet cafa1_set;
	cafa1_set.Load("/protein/work/baseline/cafa1_testset_mf.proteinset");
	
}

void Predict(const GOTermSet& go_set, const ProteinSequenceSet& train_set, const ProteinSequenceSet& test_set, const vector<BlastPredictInstance>& ori_blast_instances, const string predict_type) {
	vector<BlastPredictInstance> tmp_blast_instances;
	for (size_t i = 0; i < ori_blast_instances.size(); ++i) {
		const ProteinSequence& protein = test_set.QueryByName(ori_blast_instances[i].protein_);
		vector<int> vec_go;
		for (auto go : protein.go_terms()) {
			if (!go_set.HasKey(go.id_)) {
				continue;
			}
			if (go_set.QueryGOTerm(go.id_).type() == predict_type)
				vec_go.push_back(go.id_);
		}
		MultiLabelGoldAnswer gold_answer;
		for (int g : vec_go)
			gold_answer.insert(g);
		if (predict_type == kMF && gold_answer.size() == 1 && *gold_answer.begin() == 5515)
			continue;
		tmp_blast_instances.push_back(ori_blast_instances[i]);
	}
	clog << "Left " << tmp_blast_instances.size() << " instance" << endl;
	vector<BlastPredictInstance> blast_instances = tmp_blast_instances;
	
	vector<MultiLabelGoldAnswer> cafa1_gold_answers;
	for (size_t i = 0; i < blast_instances.size(); ++i) {
		const ProteinSequence& protein = test_set.QueryByName(blast_instances[i].protein_);
		vector<int> vec_go;
		for (auto go : protein.go_terms()) {
			if (!go_set.HasKey(go.id_)) {
				clog << "Warning: " << go.id_ << " cannot find in the go set" << endl;
				continue;
			}
			if (go_set.QueryGOTerm(go.id_).id() != go.id_)
				cout << "gold id changed" << endl;
			if (go_set.QueryGOTerm(go.id_).type() == predict_type)
				vec_go.push_back(go.id_);
//			else
//				clog << "Warning: GO:" << go.id_ << " " << go_set.QueryGOTerm(go.id_).type() << " not matched!" << endl;
		}
		MultiLabelGoldAnswer gold_answer;
		for (int g : go_set.FindAncestors(vec_go)) {
			if (go_set.HasKey(g) && go_set.QueryGOTerm(g).type() == predict_type)
				gold_answer.insert(go_set.QueryGOTerm(g).id());
		}
		cafa1_gold_answers.push_back(gold_answer);
	}
	clog << "Get " << cafa1_gold_answers.size() << " gold answers " << endl;
	
//	vector<MultiLabelPredictAnswer> predict_answers = BlastPredict(cafa1_train_set, go_set, blast_instances);
//	vector<MultiLabelPredictAnswer> predict_answers = PriorPredict(cafa1_train_set, go_set, blast_instances);
	vector<MultiLabelPredictAnswer> predict_answer;
	predict_answer = PriorPredict(train_set, go_set, blast_instances, predict_type);
	
	clog << "Begin output predict result" << endl;
//	OutputPredictResult("predict_answer_bp_tmp.csv", blast_instances, cafa1_gold_answers_bp, predict_bp);
//	OutputPredictResult("predict_answer_mf_tmp.csv", blast_instances, cafa1_gold_answers_mf, predict_mf);
	
	clog << "Begin Evaluation" << endl;
	pair<double, double> max_f_measure = GetFMeasureMax(cafa1_gold_answers, predict_answer);
	cout << "FMax = " << max_f_measure.first << endl;
	cout << "Threshold = " << max_f_measure.second << endl;
}

int main() {
//	CountGOTermSet();
//	return 0;
	
	unordered_set<string> list;
	ifstream fin("ProteinList.in");
	string line;
	while(fin >> line)
		list.insert(line);
	clog << "protein list.size = " << list.size() << endl;
	
    //Init();
	GOTermSet go_set;
	go_set.ParseGo("gene_ontology_edit.obo.2011-01-01");
	go_set.Save("go_term_set_20110101.gotermset");
	clog << "Load go term set successuful" << endl;
	
	vector<BlastPredictInstance> blast_instances = ParseBlastPredictResult("blast_cafa1_only_mf_predict.txt");
	clog << "Total load " << blast_instances.size() << " instances " << endl;

	ProteinSequenceSet cafa1_train_set;
	cafa1_train_set.Load("/protein/work/cafa1_only_species_trainset.proteinset");
	cafa1_train_set.BuildNameIndex();

	ProteinSequenceSet cafa1_set;
	cafa1_set.Load("/protein/work/baseline/cafa1_testset_mf.proteinset");
	cafa1_set.BuildNameIndex();
	Predict(go_set, cafa1_train_set, cafa1_set, blast_instances, kMF);
	return 0;
	
	vector<BlastPredictInstance> tmp_blast_instances;
	for (size_t i = 0; i < blast_instances.size(); ++i) {
		const ProteinSequence& protein = cafa1_set.QueryByName(blast_instances[i].protein_);
		vector<int> vec_bp, vec_mf;
		for (auto go : protein.go_terms()) {
			if (!go_set.HasKey(go.id_)) {
				continue;
			}
			if (go_set.QueryGOTerm(go.id_).type() == kBP)
				vec_bp.push_back(go.id_);
			else if (go_set.QueryGOTerm(go.id_).type() == kMF)
				vec_mf.push_back(go.id_);
//			else
//				clog << "Warning: GO:" << go.id_ << " " << go_set.QueryGOTerm(go.id_).type() << " not matched!" << endl;
		}
		MultiLabelGoldAnswer gold_answer_mf;
		MultiLabelGoldAnswer gold_answer_bp;
		for (int g : vec_bp)
			gold_answer_bp.insert(g);
		for (int g : vec_mf)
			gold_answer_mf.insert(g);
//		if (gold_answer_mf.size() == 1 && *gold_answer_mf.begin() == 5515)
//			continue;
		if (!list.empty() && list.count(blast_instances[i].protein_) == 0)
			continue;
		tmp_blast_instances.push_back(blast_instances[i]);
	}
	clog << "Left " << tmp_blast_instances.size() << " instance" << endl;
	blast_instances = tmp_blast_instances;
	
	vector<MultiLabelGoldAnswer> cafa1_gold_answers_mf;
	vector<MultiLabelGoldAnswer> cafa1_gold_answers_bp;
	for (size_t i = 0; i < blast_instances.size(); ++i) {
		const ProteinSequence& protein = cafa1_set.QueryByName(blast_instances[i].protein_);
		vector<int> vec_bp, vec_mf;
		for (auto go : protein.go_terms()) {
			if (!go_set.HasKey(go.id_)) {
//				vec_bp.push_back(go.id_);
//				vec_mf.push_back(go.id_);
				clog << "Warning: " << go.id_ << " cannot find in the go set" << endl;
				continue;
			}
			if (go_set.QueryGOTerm(go.id_).id() != go.id_)
				cout << "gold id changed" << endl;
			if (go_set.QueryGOTerm(go.id_).type() == kBP)
				vec_bp.push_back(go.id_);
			else if (go_set.QueryGOTerm(go.id_).type() == kMF)
				vec_mf.push_back(go.id_);
//			else
//				clog << "Warning: GO:" << go.id_ << " " << go_set.QueryGOTerm(go.id_).type() << " not matched!" << endl;
		}
		MultiLabelGoldAnswer gold_answer_mf;
		MultiLabelGoldAnswer gold_answer_bp;
		for (int g : go_set.FindAncestors(vec_bp)) {
			if (go_set.HasKey(g) && go_set.QueryGOTerm(g).type() == kBP)
				gold_answer_bp.insert(go_set.QueryGOTerm(g).id());
			else if (!go_set.HasKey(g))
				gold_answer_bp.insert(g);
		}
		for (int g : go_set.FindAncestors(vec_mf)) {
			if (go_set.HasKey(g) && go_set.QueryGOTerm(g).type() == kMF)
				gold_answer_mf.insert(go_set.QueryGOTerm(g).id());
			else if (!go_set.HasKey(g))
				gold_answer_mf.insert(g);
		}
		cafa1_gold_answers_mf.push_back(gold_answer_mf);
		cafa1_gold_answers_bp.push_back(gold_answer_bp);
	}
	clog << "Get " << cafa1_gold_answers_mf.size() << " gold answers " << endl;
	
//	vector<MultiLabelPredictAnswer> predict_answers = BlastPredict(cafa1_train_set, go_set, blast_instances);
//	vector<MultiLabelPredictAnswer> predict_answers = PriorPredict(cafa1_train_set, go_set, blast_instances);
	vector<MultiLabelPredictAnswer> predict_bp, predict_mf;
	predict_bp = PriorPredict(cafa1_train_set, go_set, blast_instances, kBP);
	predict_mf = PriorPredict(cafa1_train_set, go_set, blast_instances, kMF);
	/*
	for (size_t i = 0; i < predict_answers.size(); ++i) {
		MultiLabelPredictAnswer tmp_bp, tmp_mf;
		for (auto pr : predict_answers[i]) {
			if (!go_set.HasKey(pr.first)) {
				cerr << "Warning: GO:" << pr.first << " cannot find" << endl;
				tmp_bp.push_back(pr);
				tmp_mf.push_back(pr);
				continue;
			}
			if (go_set.QueryGOTerm(pr.first).type() == kBP)
				tmp_bp.push_back({go_set.QueryGOTerm(pr.first).id(), pr.second});
			else if (go_set.QueryGOTerm(pr.first).type() == kMF)
				tmp_mf.push_back({go_set.QueryGOTerm(pr.first).id(), pr.second});
//			else
//				clog << "Warning: GO:" << pr.first << " " << go_set.QueryGOTerm(pr.first).type() << " not matched!" << endl;
		}
//		predict_bp.push_back(GetAncestorScore(go_set, tmp_bp));
//		predict_mf.push_back(GetAncestorScore(go_set, tmp_mf));
		predict_bp.push_back(tmp_bp);
		predict_mf.push_back(tmp_mf);
	}
	*/
//	clog << "Get " << predict_answers.size() << " predict answers " << endl;
	clog << "Begin output predict result" << endl;
	OutputPredictResult("predict_answer_bp_tmp.csv", blast_instances, cafa1_gold_answers_bp, predict_bp);
	OutputPredictResult("predict_answer_mf_tmp.csv", blast_instances, cafa1_gold_answers_mf, predict_mf);
	
	clog << "Begin Evaluation" << endl;
	pair<double, double> max_f_measure = GetFMeasureMax(cafa1_gold_answers_bp, predict_bp);
	cout << "BP FMax = " << max_f_measure.first << endl;
	cout << "BP Threshold = " << max_f_measure.second << endl;
	pair<double, double> max_f_measure_mf = GetFMeasureMax(cafa1_gold_answers_mf, predict_mf);
	cout << "MF FMax = " << max_f_measure_mf.first << endl;
	cout << "MF Threshold = " << max_f_measure_mf.second << endl;
	
// 	for (auto pr : predict_bp.front())
// 		if (go_set.QueryGOTerm(pr.first).alt_ids().size() > 0)
// 			cout << "BP: " << pr.first << " " << pr.second << endl;
// 	for (auto pr : predict_mf.front())
// 		if (go_set.QueryGOTerm(pr.first).alt_ids().size() > 0)
// 			cout << "MF: " << pr.first << " " << pr.second << endl;
	clog << "Complete" << endl;
	return 0;
}
