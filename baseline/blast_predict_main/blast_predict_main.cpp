#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <fstream>
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
std::unordered_set<std::string> kExp = { "EXP","IDA","IPI","IMP","IGI","IEP","TAS","IC" };
const vector<string> kLeftSpecies = { "BACSU","PSEAE","STRPN","ARATH","YEAST","XENLA","HUMAN","MOUSE","RAT","DICDI","ECOLI" };

struct BlastPredictInstance {
	string protein_;
	vector<string> similar_proteins_;
	vector<double> similar_evalues_;
	vector<double> similar_bitscore_;

	friend boost::serialization::access;
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & protein_  & similar_proteins_ & similar_evalues_ & similar_bitscore_;
	}
};

vector<BlastPredictInstance> ParseBlastPredictResult(const std::string& filename, const ProteinSet& test_set) {
	const size_t kEvalueIndex = 10;
	const size_t kBitscoreIndex = 11;

	unordered_map<string, BlastPredictInstance> read_instances;

	ifstream fin(filename);
	string last_line;
	string line;
	while (true) {
		if (last_line.empty()) {
			while (getline(fin, line)) {
				if (line.front() != '#' && line != "Search has CONVERGED!")
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
		instance.similar_evalues_.push_back(stod(tokens[kEvalueIndex]));
		instance.similar_bitscore_.push_back(stod(tokens[kBitscoreIndex]));
		last_line.clear();
		while (getline(fin, line)) {
			if (line.front() == '#' || line == "Search has CONVERGED!")
				break;
			tokens.clear();
			split(tokens, line, is_any_of("\t"));
			string name = tokens[0];
			if (name != instance.protein_) {
				last_line = line;
				break;
			}
			instance.similar_proteins_.push_back(tokens[1]);
			instance.similar_evalues_.push_back(stod(tokens[kEvalueIndex]));
			instance.similar_bitscore_.push_back(stod(tokens[kBitscoreIndex]));
			//cout << instance.similar_evalues_.back() << ", " << instance.similar_bitscore_.back() << endl;
		}
		
		read_instances[instance.protein_] = instance;
	}
	vector<BlastPredictInstance> ret_instances;
	for (int i = 0; i < test_set.Size(); ++i) {
		string id = test_set[i].id_;
		if (read_instances.count(id) > 0) {
			ret_instances.push_back(read_instances.at(id));
		}
		else {
			BlastPredictInstance tmp;
			tmp.protein_ = id;
			ret_instances.push_back(tmp);
			clog << "Error: " << id << " cannot found blast prediction" << endl;
		}
	}
	return ret_instances;
}

vector<BlastPredictInstance> ParseBlastPredictXmlResult(const std::string& xml_file, const ProteinSet& test_set) {
	using namespace boost::property_tree;
	vector<BlastPredictInstance> ret_instances;
	/*
	ptree rt_tree;
	try {
		read_xml(xml_file, rt_tree);
	}
	catch (const std::exception& e) {
		cerr << "Error: " << e.what() << endl;
		return{};
	}

	ptree iter_tree = rt_tree.get_child("BlastOutput.BlastOutput_iterations");

	clog << "Begin parse xml result" << endl;
	unordered_map<string, BlastPredictInstance> read_instances;
	BOOST_FOREACH(const ptree::value_type& u, iter_tree) {
		if (u.first == "Iteration") {
			BlastPredictInstance instance;
			instance.protein_ = u.second.get<string>("Iteration_query-def");
			BOOST_FOREACH(const ptree::value_type& v, u.second.get_child("Iteration_hits")) {
				if (v.first == "Hit") {
					string neibo_id = v.second.get<string>("Hit_def");
					double bit_score = v.second.get<double>("Hit_hsps.Hsp.Hsp_bit-score");
					double evalue = v.second.get<double>("Hit_hsps.Hsp.Hsp_evalue");
					instance.similar_proteins_.push_back(neibo_id);
					instance.similar_bitscore_.push_back(bit_score);
					instance.similar_evalues_.push_back(evalue);
				}
			}
			read_instances[instance.protein_] = instance;
			if (read_instances.size() % 200 == 0) {
				clog << read_instances.size() << " xml result parsed" << endl;
			}
		}
	}

	
	for (int i = 0; i < test_set.Size(); ++i) {
		string id = test_set[i].id_;
		if (read_instances.count(id) > 0) {
			ret_instances.push_back(read_instances.at(id));
		}
		else {
			BlastPredictInstance tmp;
			tmp.protein_ = id;
			ret_instances.push_back(tmp);
			clog << "Error: " << id << " cannot found blast prediction" << endl;
		}
	}

	ofstream fout("yrh_xml_result.bin", ios_base::binary);
	boost::archive::binary_oarchive oa(fout);
	oa << ret_instances;
	fout.close();
	*/
	ifstream fin("yrh_xml_result.bin", ios_base::binary);
	boost::archive::binary_iarchive ia(fin);
	ia >> ret_instances;
	fin.close();

	return ret_instances;
}

double EvalueToScore(double evalue) {
	if (evalue > 1e-250) {
		return -log10(evalue);
	}
	return 250;
}

MultiLabelPredictAnswer BlastPredict(const ProteinSet& train_set, const GOTermSet& goterm_set, const BlastPredictInstance& test_instance, GoType go_type) {
	unordered_map<int, double> label_score;
	for (size_t i = 0; i < test_instance.similar_proteins_.size(); ++i) {
		if (train_set.Has(test_instance.similar_proteins_[i])) {
			const Protein &protein = train_set[test_instance.similar_proteins_[i]];
			/*if (test_instance.protein_ == "Q7K0A0" && go_type == MF) {
				clog << "Q7K0A0 Neibo: " << test_instance.similar_proteins_[i] << endl;
				clog << "Indexed go: " << protein.go_term(go_type).size() << endl;
			}*/
			double score = EvalueToScore(test_instance.similar_evalues_[i]);
			vector<int> go_leaves = protein.go_term(go_type);
			for (int go_id : goterm_set.FindAncestors(go_leaves)) {
				if (label_score.count(go_id) == 0)
					label_score[go_id] = score;
				else
					label_score[go_id] = max(label_score[go_id], score);
			}
		}
		else
			clog << "Warning: " << test_instance.similar_proteins_[i] << " cannot found in trainset" << endl;
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
	for (int i = 0; i < test_instances.size(); ++i) {
		predict_answers[i] = BlastPredict(train_set, goterm_set, test_instances[i], go_type);
	}
	
	return predict_answers;
}

vector<MultiLabelGoldAnswer> GetGroundTruth(const GOTermSet& goterm_set, const ProteinSet& test_set, GoType go_type) {
	vector<MultiLabelGoldAnswer> ret;
	for (int i = 0; i < test_set.Size(); ++i) {
		vector<int> gos = goterm_set.FindAncestors(test_set[i].go_term(go_type));
		ret.push_back({gos.begin(), gos.end()});
	}
	return ret;
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

GoType GoTypeStrToTypeId(string type_str) {
	for (int i = MF; i < GO_TYPE_SIZE; ++i)
		if (kGoTypeStr[i] == type_str)
			return (GoType)i;
	return GO_TYPE_SIZE;
}

void AlignEvaluation() {
	const string kWorkDir = "D:/workspace/cafa/work/";
	const string kGoTermSetFile = kWorkDir + "go_160601.gotermset";
	const string kBlastPredictFile = kWorkDir + "group1_test_blast_iter3.txt";
	const string kTrainProteinSetFile = kWorkDir + "cafa3_train_161222.proteinset";
	const string kTestProteinSetFile = kWorkDir + "cafa3_test_161222.proteinset";
	const string kBlastResultFile = kWorkDir + "blast.res";

	GOTermSet go_set;
	go_set.Load(kGoTermSetFile);

	ProteinSet test_set;
	test_set.Load(kTestProteinSetFile);

	unordered_map<string, vector<MultiLabelPredictAnswer>> blast_predictions;
	ifstream fin(kBlastResultFile);
	string line;
	for (int i = 0; i < 3; ++i)
		getline(fin, line);
	int cnt = 0;
	while (getline(fin, line)) {
		if (line == "END")
			break;
		vector<string> tokens;
		split(tokens, line, boost::is_any_of("\t "));
		if (blast_predictions.count(tokens[0]) == 0)
			blast_predictions[tokens[0]].resize(GO_TYPE_SIZE);
		
		int go_id = stoi(tokens[1].substr(3));
		double score = stod(tokens[2]);
		if (go_set.HasKey(go_id)) {
			GoType type = GoTypeStrToTypeId(go_set.QueryGOTerm(go_id).type());
			if (type == GO_TYPE_SIZE)
				cerr << "Error: cannot found go type for " << go_id << endl;
			blast_predictions[tokens[0]][type].push_back({ go_id, score });
			++cnt;
		}
	}
	clog << "Total load " << cnt << " scores" << endl;

	for (int go_type = MF; go_type < GO_TYPE_SIZE; ++go_type) {
		vector<MultiLabelGoldAnswer> gold_standard = GetGroundTruth(go_set, test_set, (GoType)(go_type));
		vector<MultiLabelPredictAnswer> prediction;
		for (int i = 0; i < test_set.Size(); ++i) {
			if (blast_predictions.count(test_set[i].id_) > 0) {
				prediction.push_back(blast_predictions[test_set[i].id_][go_type]);
				sort(prediction.back().begin(), prediction.back().end(), [](const pair<int, double>& p1, const pair<int, double>& p2) {return p1.second > p2.second; });
			}
			else
				prediction.push_back({});
		}

		int indexed_cnt = 0;
		for (int i = 0; i < gold_standard.size(); ++i)
			if (!gold_standard[i].empty())
				++indexed_cnt;
		clog << "Total " << indexed_cnt << " protein indexed" << endl;
		pair<double, double> eva = GetFMeasureMax(gold_standard, prediction);
		clog << kGoTypeStr[go_type] << " FMax: " << eva.second << ", Thres: " << eva.first << endl;
		//pair<double, double> eva_old = GetFMeasureMaxOld(gold_standard, prediction);
		//clog << kGoTypeStr[go_type] << " Old FMax: " << eva_old.first << ", Old Thres: " << eva_old.second << endl;
	}
	system("pause");
}

int main() {
	//const string kWorkDir = "D:/workspace/cafa/work/";
	const string kWorkDir = "./";
	const string kGoTermSetFile = kWorkDir + "go_160601.gotermset";
	const string kBlastPredictFile = kWorkDir + "group1_test_blast_iter3.txt";
	const string kTrainProteinSetFile = kWorkDir + "cafa3_train_161222.proteinset";
	const string kTestProteinSetFile = kWorkDir + "cafa3_test_161222.proteinset";

	//AlignEvaluation();
	//return 0;

	GOTermSet go_set;
	go_set.Load(kGoTermSetFile);

	//go_set.ParseGo("C:/psw/cafa/CAFA3/Ontology/gene_ontology_edit.obo.2016-06-01");

	ProteinSet train_set;
	train_set.Load(kTrainProteinSetFile);
	ProteinSet test_set;
	test_set.Load(kTestProteinSetFile);

	//vector<BlastPredictInstance> blast_result = ParseBlastPredictResult(kBlastPredictFile, test_set);
	vector<BlastPredictInstance> blast_result = ParseBlastPredictXmlResult(kWorkDir + "a5a6187809ace29f9a9100619c0c94a3.xml", test_set);
	clog << "Total load " << blast_result.size() << " blast results" << endl;

	for (int go_type = MF; go_type < GO_TYPE_SIZE; ++go_type) {
		vector<MultiLabelGoldAnswer> gold_standard = GetGroundTruth(go_set, test_set, (GoType)(go_type));
		vector<MultiLabelPredictAnswer> prediction = BlastPredict(train_set, go_set, blast_result, (GoType)go_type);

		OutputPredictResult(kGoTypeStr[go_type] + ".txt", blast_result, gold_standard, prediction);

		int indexed_cnt = 0;
		for (int i = 0; i < gold_standard.size(); ++i)
			if (!gold_standard[i].empty())
				++indexed_cnt;
		clog << "Total " << indexed_cnt << " protein indexed" << endl;
		pair<double, double> eva = GetFMeasureMax(gold_standard, prediction);
		clog << kGoTypeStr[go_type] << " FMax: " << eva.second  << ", Thres: " << eva.first << endl;

		//pair<double, double> eva_old = GetFMeasureMaxOld(gold_standard, prediction);
		//clog << kGoTypeStr[go_type] << " Old FMax: " << eva_old.first << ", Old Thres: " << eva_old.second << endl;
	}

	system("pause");
	return 0;
}
