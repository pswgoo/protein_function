#pragma once

#include <vector>
#include <unordered_set>

typedef std::vector<std::pair<int,double>> MultiLabelPredictAnswer;
typedef std::unordered_set<int> MultiLabelGoldAnswer;

/**
 * @TODO test the function
 * @return <threshold, max-FMeasure>
 */
std::pair<double,double> GetFMeasureMax(const std::vector<MultiLabelGoldAnswer>& gold_standard, const std::vector<MultiLabelPredictAnswer>& predict_answser);

std::pair<double, double> GetFMeasureMaxOld(const std::vector<MultiLabelGoldAnswer>& gold_standard, const std::vector<MultiLabelPredictAnswer>& predict_answser);

double GetAverageAUC(const std::vector<MultiLabelGoldAnswer>& gold_standard, const std::vector<MultiLabelPredictAnswer>& predict_answser);
