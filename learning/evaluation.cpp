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
 * @return <max-FMeasure, threshold>
 */
std::pair<double, double> GetFMeasureMax(const std::vector<MultiLabelGoldAnswer>& gold_standard, const std::vector<MultiLabelPredictAnswer>& predict_answser) {
	pair<double, double> ret(-1, -1);
	for (double t = 0.01; t <= 1.0 + EPS; t += 0.01) {
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
					sum_pre += tp_cnt / pos_cnt;
				}
				sum_rec += tp_cnt / gold_standard[i].size();
			}
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
