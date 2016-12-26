#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <unordered_map>

#include <boost/algorithm/string.hpp>

#include "data_class/go_term.h"
#include "data_class/protein_sequence.h"

using namespace std;

int main() {

	GOTermSet go_term_set;
	go_term_set.ParseGo("D:/workspace/cafa/CAFA3/Ontology/gene_ontology_edit.obo.2016-06-01");
	clog << "Total load " << go_term_set.go_terms().size() << " GOs" << endl;
	go_term_set.Save("go_160601.gotermset");

	// sort()

	return 0;

	ProteinSet train_set;
	train_set.ParseRawTxt("D:/workspace/cafa/CAFA3/train.fasta", "D:/workspace/cafa/CAFA3/train_mf.txt", "D:/workspace/cafa/CAFA3/train_bp.txt", "D:/workspace/cafa/CAFA3/train_cc.txt", false);
	train_set.Save("cafa3_train_161222.proteinset");

	ProteinSet test_set;
	test_set.ParseRawTxt("D:/workspace/cafa/CAFA3/group1/test.fasta", "D:/workspace/cafa/CAFA3/test_mf.txt", "D:/workspace/cafa/CAFA3/test_bp.txt", "D:/workspace/cafa/CAFA3/test_cc.txt");
	test_set.Save("cafa3_test_161222.proteinset");

	system("pause");
    return 0;
}
