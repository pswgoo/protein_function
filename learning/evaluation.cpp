#include "evaluation.h"

#include <assert.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include "common/common_basic.h"

using namespace std;

/**
 * @TODO test the function
 * @return <threshold, max-FMeasure>
 */
std::pair<double, double> GetFMeasureMax(const std::vector<MultiLabelGoldAnswer>& gold_standard, const std::vector<MultiLabelPredictAnswer>& predict_answser) {
	double mn = numeric_limits<double>::max(), mx = numeric_limits<double>::lowest();
	for (const MultiLabelPredictAnswer& answer : predict_answser)
		for (const pair<int, double>& pr : answer) {
			mx = max(mx, pr.second);
			mn = min(mn, pr.second);
		}

	double dist = (mx - mn) / 100.0;
	//mn = 0.0, mx = 1.0, dist = 0.01;
	pair<double, double> ret(-1e10, -1);
	for (double t = mn; t <= mx + EPS; t += dist) {
		int m = 0;
		int n = 0;
		double sum_pre = 0.0, sum_rec = 0.0;
		for (int i = 0; i < predict_answser.size(); ++i)
			if (gold_standard[i].size() > 0) {
				++n;
				int pos_cnt = 0;
				int tp_cnt = 0;
				for (pos_cnt = 0; pos_cnt < predict_answser[i].size() && predict_answser[i][pos_cnt].second >= t; ++pos_cnt)
					if (gold_standard[i].count(predict_answser[i][pos_cnt].first) > 0)
						++tp_cnt;
				if (pos_cnt) {
					++m;
					//cout << tp_cnt << " : " << pos_cnt << endl;
					sum_pre += double(tp_cnt) / pos_cnt;
				}
				sum_rec += double(tp_cnt) / gold_standard[i].size();
			}
		//cout << t << " " << sum_pre << " " << m << " " << sum_rec << " " << n << endl;
		double pr = sum_pre / m;
		double rc = sum_rec / n;
		if (pr + rc > 0) {
			double f = 2 * pr * rc / (pr + rc);
			if (f > ret.second)
				ret = { t, f };
		}
	}
	return ret;
}

/**
* @TODO test the function
* @return <max-FMeasure, threshold>
*/
std::pair<double, double> GetFMeasureMaxOld(const std::vector<MultiLabelGoldAnswer>& gold_standard, const std::vector<MultiLabelPredictAnswer>& predict_answser) {
	assert(gold_standard.size() == predict_answser.size());
	pair<double, double> ret_value;

	struct PredictNode {
		double score_;
		double label_id_;
		size_t instance_id_;
		bool is_true_;
	};

	vector<PredictNode> vec_nodes;
	for (size_t i = 0; i < gold_standard.size(); ++i) {
		if (gold_standard[i].empty())
			continue;
		for (const auto& u : predict_answser[i]) {
			PredictNode node;
			node.score_ = u.second;
			node.label_id_ = u.first;
			node.instance_id_ = i;
			node.is_true_ = (gold_standard[i].count(u.first) > 0);
			vec_nodes.push_back(node);
		}
	}

	sort(vec_nodes.begin(), vec_nodes.end(), [](const PredictNode& lhs, const PredictNode& rhs) { return lhs.score_ > rhs.score_; });

	double sum_precision = 0.0;
	double sum_recall = 0.0;
	unordered_set<size_t> pos_pre_instances;
	vector<int> tp(gold_standard.size());
	vector<int> pos(gold_standard.size());
	for (size_t i = 0; i < tp.size(); ++i)
		tp[i] = pos[i] = 0;

	int has_indexed_count = 0;
	for (size_t i = 0; i < gold_standard.size(); ++i)
		if (gold_standard[i].size() > 0)
			has_indexed_count++;
	if (!has_indexed_count)
		return{ 0, 0 };
	ret_value.first = 0.0;
	ret_value.second = vec_nodes[0].score_ + EPS;

	size_t cur = 0;
	while (cur < vec_nodes.size()) {
		double thres = vec_nodes[cur].score_ - EPS;
		unordered_map<size_t, int> update_pos;
		unordered_map<size_t, int> update_tp;
		while (cur < vec_nodes.size() && thres <= vec_nodes[cur].score_) {
			if (update_pos.count(vec_nodes[cur].instance_id_) == 0) {
				update_pos[vec_nodes[cur].instance_id_] = 0;
				update_tp[vec_nodes[cur].instance_id_] = 0;
			}

			++update_pos[vec_nodes[cur].instance_id_];
			pos_pre_instances.insert(vec_nodes[cur].instance_id_);
			if (vec_nodes[cur].is_true_) {
				++update_tp[vec_nodes[cur].instance_id_];
			}
			++cur;
		}

		for (auto it1 = update_pos.begin(), it2 = update_tp.begin(); it1 != update_pos.end() && it2 != update_tp.end(); ++it1, ++it2) {
			if (it1->first != it2->first)
				cerr << "Error: evaluation error " << it1->first << " != " << it2->first << endl;
			size_t ins_id = it1->first;
			double old_pre = 0.0, old_rec = 0.0;
			if (pos[ins_id] > 0) {
				old_pre = double(tp[ins_id]) / pos[ins_id];
			}
			if (gold_standard[ins_id].size() > 0) {
				old_rec = double(tp[ins_id]) / gold_standard[ins_id].size();
			}

			tp[ins_id] += it2->second;
			pos[ins_id] += it1->second;
			double nw_pre = 0.0, nw_rec = 0.0;
			if (pos[ins_id] > 0) {
				nw_pre = double(tp[ins_id]) / pos[ins_id];
			}
			if (gold_standard[ins_id].size() > 0) {
				nw_rec = double(tp[ins_id]) / gold_standard[ins_id].size();
			}
			sum_precision = sum_precision - old_pre + nw_pre;
			sum_recall = sum_recall - old_rec + nw_rec;
		}

		if (pos_pre_instances.size() > 0) {
			//			double pre = sum_precision / pos_pre_instances.size();
			//			double rec = gold_standard.size() > 0 ?  sum_recall / gold_standard.size() : 0;
			double pre = sum_precision / min(has_indexed_count, (int)pos_pre_instances.size());
			double rec = has_indexed_count > 0 ? sum_recall / has_indexed_count : 0;
			double fm = (pre + rec) < EPS ? 0.0 : 2 * pre * rec / (pre + rec);
			if (fm > ret_value.first) {
				ret_value.first = fm;
				ret_value.second = thres;
			}
		}
	}
	clog << "Evaluated " << has_indexed_count << " indexed proteins" << endl;
	return ret_value;
}
