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
	const string kWorkDir = "C:/psw/cafa/CAFA3/"; // C:/psw/cafa/CAFA3/work/

	GOTermSet go_term_set;
	go_term_set.ParseGo(kWorkDir + "Ontology/gene_ontology_edit.obo.2016-06-01");
	clog << "Total load " << go_term_set.go_terms().size() << " GOs" << endl;
	go_term_set.Save(kWorkDir + "work/go_160601.gotermset");

	// sort()

	//return 0;

	ProteinSet train_set;
	train_set.ParseRawTxt(kWorkDir + "train.fasta", kWorkDir + "train_mf.txt", kWorkDir + "train_bp.txt", kWorkDir + "train_cc.txt", false);
	train_set.Save(kWorkDir + "work/cafa3_train_161222.proteinset");

	ProteinSet test_set;
	test_set.ParseRawTxt(kWorkDir + "group1/test.fasta", kWorkDir + "test_mf.txt", kWorkDir + "test_bp.txt", kWorkDir + "test_cc.txt");
	test_set.Save(kWorkDir + "work/cafa3_test_161222.proteinset");

	system("pause");
    return 0;
}
