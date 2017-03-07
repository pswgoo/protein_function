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
const int kAminoTypeSize = 20;

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
			if (type != NON) {
				int type_idx = type - 1;
				output_data[offset + type_idx * kMaxSequenceLength + j] = 1.0;
			}
		}
		vector<int> go_ids = go_set.FindAncestors(proteins[indices[i]].all_go_terms());
		for (int go : go_ids) {
			int idx = go_set.GetIndex(go);
			if (idx >= 0)
				output_label[i * nr_class + idx] = 1.0f;
		}
	}
}

Expression Model(Expression t, int nr_class, bool is_train = false) {
	t = Reshape(t, { 20, 1, 1000 });
	t = ReLU(ConvolutionLayer("conv1", t, 40, Shape(1, 5), Shape(1, 1), Shape(0, 0)));
	t = Reshape(t, { 40, 1, 996 });
	t = MaxPooling(t, Shape(1, 2), Shape(1, 2), Shape(0, 0));
	t = Reshape(t, { 40, 1, 498 });
	t = ReLU(ConvolutionLayer("conv2", t, 80, Shape(1, 5), Shape(1, 1), Shape(0)));
	t = MaxPooling(t, Shape(1, 2), Shape(1, 2), Shape(0, 0));
	t = Reshape(t, Shape(t.GetShape().GetSize()));
	t = ReLU(LinearLayer("l1", t, 500));
	t = Sigmoid(LinearLayer("l2", t, nr_class));
	return t;
}

int main() {
	//const string kWorkDir = "C:/psw/cafa/protein_cafa2/work/";
	const string kWorkDir = "./"; 

	const string kGoTermSetFile = kWorkDir + "go_140101.gotermset";
	const string kTrainProteinSetFile = kWorkDir + "cafa2_train_170307.proteinset";
	const string kTestProteinSetFile = kWorkDir + "cafa2_test_170307.proteinset";

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

	const int kEpochNumber = 20;
	int batch_size = 1;

	Device device(GPU);
	Graph graph(&device);
	graph.SetRandomSeed(0);
	AdamOptimizer optimizer(&graph);

	vector<float> train_batch_data;
	vector<float> train_batch_label;

	double total_loss = 0;
	int nr_class = (int)go_set.go_terms().size();

	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
	for (int e = 0; e < kEpochNumber; ++e) {
		total_loss = 0;
		vector<int> indices(train_set.Size(), 0);
		iota(indices.begin(), indices.end(), 0);
		for (int b = 0; b < indices.size(); b += batch_size) {
			if (b % 2000 == 0 && b != 0) {
				clog << "Epoch " << e << " batch " << b << " Loss: " << total_loss / b << endl;

				vector<float> test_batch_data;
				vector<float> test_batch_label;
				vector<MultiLabelPredictAnswer> prediction[GO_TYPE_SIZE];
				for (int i = 0; i < GO_TYPE_SIZE; ++i)
					prediction[i].resize(test_set.Size());
				for (int i = 0; i < test_set.Size(); ++i) {
					ExtractBatch(go_set, test_set, { i }, nr_class, test_batch_data, test_batch_label);
					Expression t = Input(&graph, Shape(kAminoTypeSize, kMaxSequenceLength), test_batch_data.data());
					t = Model(t, nr_class);
					vector<float> ans(nr_class);
					t.Forward().GetValue(ans.data());
					for (int j = 0; j < nr_class; ++j) {
						if (ans[j] > 1e-3) {
							const GOTerm & go_term = go_set.go_terms()[j];
							prediction[GoTypeStrToTypeId(go_term.type())][i].push_back({ go_term.id(), (double)ans[j] });
						}
					}
				}
				for (int i = MF; i < GO_TYPE_SIZE; ++i) {
					pair<double, double> eva = GetFMeasureMax(ground_truth[i], prediction[i]);
					clog << "Epoch: " << e << " " << kGoTypeStr[i] << " FMax: " << eva.second << ", Thres: " << eva.first << endl;
				}
			}
			int len = b + batch_size >= indices.size() ? (int)indices.size() - b : batch_size;
			ExtractBatch(go_set, train_set, { indices.begin() + b, indices.begin() + b + len }, nr_class, train_batch_data, train_batch_label);
			Expression t = BatchInput(&graph, len, Shape(kAminoTypeSize, kMaxSequenceLength), train_batch_data.data());
			t = Model(t, nr_class, true);

			Expression label = BatchInput(&graph, len, Shape(nr_class), train_batch_label.data());
			Expression loss = ReduceSum(BinaryCrossEntropy(t, label));

			total_loss += loss.Forward().ReduceSum();

			loss.Forward();
			loss.Backward();
			optimizer.Update();
		}

		std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
		std::cout << e << " Loss: " << total_loss / train_set.Size();

	}

	clog << "Complete" << std::endl;
	return 0;
}
