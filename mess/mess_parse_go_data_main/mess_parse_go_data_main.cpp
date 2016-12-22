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
	go_term_set.ParseGo("C:/psw/cafa/CAFA3/Ontology/gene_ontology_edit.obo.2016-06-01");
	clog << "Total load " << go_term_set.go_terms().size() << " GOs" << endl;
	go_term_set.Save("go_160601.gotermset");

	ProteinSet train_set;
	train_set.ParseRawTxt("C:/psw/cafa/CAFA3/train.fasta", "C:/psw/cafa/CAFA3/train_mf.txt", "C:/psw/cafa/CAFA3/train_bp.txt", "C:/psw/cafa/CAFA3/train_cc.txt");
	train_set.Save("cafa3_train_161222.proteinset");

	ProteinSet test_set;
	test_set.ParseRawTxt("C:/psw/cafa/CAFA3/group1/test.fasta", "C:/psw/cafa/CAFA3/test_mf.txt", "C:/psw/cafa/CAFA3/test_bp.txt", "C:/psw/cafa/CAFA3/test_cc.txt");
	test_set.Save("cafa3_test_161222.proteinset");

	system("pause");
    return 0;
}
