#include "graynet/graynet.h"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>
#include <numeric>

#include "data_class/protein_sequence.h"
#include "data_class/go_term.h"
#include "learning/evaluation.h"

using namespace std;

const int kMaxSequenceLength = 1000;
const int kAminoTypeSize = 21;

void ExtractBatch(const GOTermSet& go_set, const ProteinSet& proteins, const vector<int>& indices, const int nr_class, vector<float>& output_data, vector<float>& output_label) {
	output_data.clear();
	output_label.clear();
	output_data.resize(indices.size() * kAminoTypeSize * kMaxSequenceLength, 0.0f);
	output_label.resize(indices.size() * nr_class, 0.0f);

	for (int i = 0; i < indices.size(); ++i) {
		const string &seq = proteins[indices[i]].sequence_;
		int offset = i * kAminoTypeSize * kMaxSequenceLength;
		for (int j = 0; j < seq.size() && j < kMaxSequenceLength; ++j) {
			AminoType type = GetAminoType(string(1, seq[j]));
			output_data[offset + type * kMaxSequenceLength + j] = 1.0;
		}
		vector<int> go_ids = go_set.FindAncestors(proteins[indices[i]].all_go_terms());
		for (int go : go_ids) {
			int idx = go_set.GetIndex(go);
			/*if (idx >= 0)
				output_label[i * nr_class + idx] = 1.0f;*/
			//if (go == 5737) // 16323
			//	output_label[i * nr_class] = 1;
		}
		
		output_label[i * nr_class] = (seq.find("LALL") != string::npos || seq.find("QQLL") != string::npos || seq.find("SSP") != string::npos);
		//output_label[i * nr_class] = count(seq.begin(), seq.end(), 'C') > 15; //seq.find("ARR") != string::npos;
	}
}

Expression Model(Expression t, int nr_class, bool is_train = false) {
	t = Reshape(t, { 1, kAminoTypeSize, kMaxSequenceLength });
	t = ReLU(ConvolutionLayer("conv1", t, 40, Shape(kAminoTypeSize, 5), Shape(1, 1), Shape(0, 0)));
	t = MaxPooling(t, Shape(1, 2), Shape(1, 2), Shape(0, 0));// 446
	t = ReLU(ConvolutionLayer("conv2", t, 80, Shape(1, 5), Shape(1, 1), Shape(0, 0)));
	t = MaxPooling(t, Shape(1, 2), Shape(1, 2), Shape(0, 0)); // 219
	t = ReLU(ConvolutionLayer("conv3", t, 160, Shape(1, 5), Shape(1, 1), Shape(0, 0)));
	t = MaxPooling(t, Shape(1, 2), Shape(1, 2), Shape(0, 0)); // 106
	t = ReLU(ConvolutionLayer("conv4", t, 320, Shape(1, 5), Shape(1, 1), Shape(0, 0)));
	t = MaxPooling(t, Shape(1, 2), Shape(1, 2), Shape(0, 0)); // 49
	//t = ReLU(ConvolutionLayer("conv5", t, 320, Shape(1, 5), Shape(1, 1), Shape(0, 0)));
	//t = MaxPooling(t, Shape(1, 2), Shape(1, 1), Shape(0, 0)); // 49
	t = Flatten(t);
	t = ReLU(LinearLayer("l1", t, 200));
	//if (is_train)
	//	t = Dropout(t, 0.5);
	t = Sigmoid(LinearLayer("l2", t, nr_class));
	return t;
}

void GetPredictAnswer(const GOTermSet& go_set, const vector<float>& label_scores, MultiLabelPredictAnswer& mf_ans, MultiLabelPredictAnswer& bp_ans, MultiLabelPredictAnswer& cc_ans) {
	vector<pair<int, double>> nw_scores = go_set.ScoreAncestors(label_scores, 1e-3);
	for (int i = 0; i < nw_scores.size(); ++i)
		if (go_set.QueryGOTerm(nw_scores[i].first).type() == kGoTypeStr[MF])
			mf_ans.push_back(nw_scores[i]);
		else if (go_set.QueryGOTerm(nw_scores[i].first).type() == kGoTypeStr[BP])
			bp_ans.push_back(nw_scores[i]);
		else if (go_set.QueryGOTerm(nw_scores[i].first).type() == kGoTypeStr[CC])
			cc_ans.push_back(nw_scores[i]);
		else
			cerr << "Error, not found go " << nw_scores[i].first << endl;
}

void Debug(const GOTermSet& go_set, const ProteinSet& test_set) {
	vector<float> test_batch_data;
	vector<float> test_batch_label;

	for (int i = 0; i < 5; ++i) {
		ExtractBatch(go_set, test_set, { i }, 1, test_batch_data, test_batch_label);
		clog << i << "label size: " << test_batch_label.size() << ", label: " << test_batch_label[0] << endl;
		/*for (int j = 0; j < kAminoTypeSize; ++j) {
			string file_name = "data/" + to_string(i) + "_" + Get1LetterAminoName((AminoType)j) + ".txt";
			ofstream fout(file_name);
			for (int k = 0; k < kMaxSequenceLength; ++k)
				fout << (int)test_batch_data[j * kMaxSequenceLength + k];
		}*/
	}
}

int main() {
	//const string kWorkDir = "C:/psw/cafa/protein_cafa2/work/";
	const string kWorkDir = "./";
/*
	const string kGoTermSetFile = kWorkDir + "go_140101.gotermset";
	const string kTrainProteinSetFile = kWorkDir + "cafa2_train_170307.proteinset";
	const string kTestProteinSetFile = kWorkDir + "cafa2_test_170307.proteinset";
*/
	const string kGoTermSetFile = kWorkDir + "go_160601.gotermset";
	const string kTrainProteinSetFile = kWorkDir + "cafa3_train_161222.proteinset";
	const string kTestProteinSetFile = kWorkDir + "cafa3_test_161222.proteinset";

	GOTermSet go_set;
	go_set.Load(kGoTermSetFile);

	ProteinSet train_set;
	train_set.Load(kTrainProteinSetFile);
	ProteinSet test_set;
	test_set.Load(kTestProteinSetFile);

	vector<MultiLabelGoldAnswer> ground_truth[GO_TYPE_SIZE];

	for (int go_type = MF; go_type < GO_TYPE_SIZE; ++go_type)
		for (int i = 0; i < test_set.Size(); ++i) {
			vector<int> gos = go_set.FindAncestors(test_set[i].go_term(GoType(go_type)));
			ground_truth[go_type].emplace_back(gos.begin(), gos.end());
		}

	const int kEpochNumber = 30;
	int batch_size = 1;

	Device device(GPU);
	Graph graph(&device);
	graph.SetRandomSeed(0);
	SGDOptimizer optimizer(&graph, 0.01f);

	vector<float> train_batch_data;
	vector<float> train_batch_label;

	double total_loss = 0;
	int nr_class = 1; //(int)go_set.go_terms().size();
	double acc_sum = 0.0;
	for (int e = 0; e < kEpochNumber; ++e) {
		total_loss = 0;
		acc_sum = 0;
		vector<int> indices(train_set.Size(), 0);
		iota(indices.begin(), indices.end(), 0);
		for (int b = 0; b < indices.size(); b += batch_size) {
			if (b % 4000 == 0 && b != 0) {
				cout << "Epoch " << e << " batch " << b << " Loss: " << total_loss / b  << ", Acc:" << acc_sum / b << endl;
				vector<float> test_batch_data;
				vector<float> test_batch_label;

				int tp = 0, tn = 0;
				float sum_ans = 0;
				float mn_ans = 1000;
				float mx_ans = -1;
				for (int i = 0; i < test_set.Size(); ++i) {
					ExtractBatch(go_set, test_set, { i }, nr_class, test_batch_data, test_batch_label);
					Expression t = BatchInput(&graph, 1, Shape(kAminoTypeSize, kMaxSequenceLength), test_batch_data.data());
					t = Model(t, nr_class);
					vector<float> ans(nr_class);
					t.Forward().GetValue(ans.data());
					//int p = rand() % 10;
					if (test_batch_label[0] > 0.5 && ans[0] > 0.5)
						++tp;
					else if (test_batch_label[0] < 0.5 && ans[0] < 0.5)
						++tn;
					sum_ans += ans[0];
					mn_ans = min(mn_ans, ans[0]);
					mx_ans = max(mx_ans, ans[0]);
				}
				clog << "Accuracy: " << float(tp + tn) / test_set.Size() << ", avg ans: " << sum_ans / test_set.Size() << ", mn_ans:" << mn_ans << ", mx_ans: "<< mx_ans << endl;
/*
				vector<float> test_batch_data;
				vector<float> test_batch_label;
				vector<MultiLabelPredictAnswer> prediction[GO_TYPE_SIZE];
				for (int go_type = 0; go_type < GO_TYPE_SIZE; ++go_type)
					prediction[go_type].resize(test_set.Size());
				for (int i = 0; i < test_set.Size(); ++i) {
					ExtractBatch(go_set, test_set, { i }, nr_class, test_batch_data, test_batch_label);
					Expression t = BatchInput(&graph, 1, Shape(kAminoTypeSize, kMaxSequenceLength), test_batch_data.data());
					t = Model(t, nr_class);
					vector<float> ans(nr_class);
					t.Forward().GetValue(ans.data());
					GetPredictAnswer(go_set, ans, prediction[MF][i], prediction[BP][i], prediction[CC][i]);
				}
				for (int i = MF; i < GO_TYPE_SIZE; ++i) {
					pair<double, double> eva = GetFMeasureMax(ground_truth[i], prediction[i]);
					cout << "Epoch: " << e << " " << kGoTypeStr[i] << " FMax: " << eva.second << ", Thres: " << eva.first << endl;
					clog << "Epoch: " << e << " " << kGoTypeStr[i] << " FMax: " << eva.second << ", Thres: " << eva.first << endl;
				}*/
			}
			int len = (b + batch_size >= indices.size()) ? ((int)indices.size() - b) : batch_size;
			ExtractBatch(go_set, train_set, { indices.begin() + b, indices.begin() + b + len }, nr_class, train_batch_data, train_batch_label);
			Expression t = BatchInput(&graph, len, Shape(kAminoTypeSize, kMaxSequenceLength), train_batch_data.data());
			t = Model(t, nr_class, true);

			Expression label = BatchInput(&graph, len, Shape(nr_class), train_batch_label.data());
			Expression loss = ReduceSum(BinaryCrossEntropy(t, label));

			float acc = BinaryClassificationAccuracy(t, label).Forward().ReduceSum();
			acc_sum += acc;
			total_loss += loss.Forward().ReduceSum();

			loss.Forward();
			loss.Backward();
			optimizer.Update();
		}

		std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
		std::cout << e << " Loss: " << total_loss / train_set.Size() << endl;
		graph.Save(("epoch" + to_string(e) + ".model").c_str());
	}

	clog << "Complete" << std::endl;
	return 0;
}
