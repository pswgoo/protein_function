#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <unordered_map>

#include <boost/algorithm/string.hpp>

#include "data_class/go_term.h"
#include "data_class/protein_sequence.h"
#include "data_class/protein_profile.h"

using namespace std;

void CAFA3() {
	const string kWorkDir = "C:/psw/cafa/CAFA3/"; // C:/psw/cafa/CAFA3/work/

	ProteinProfileSet profile_set;

	profile_set.ParseUniprotXml("uniprot_sprot_201610.xml");
	profile_set.Save("uniprot_sprot_201610.profileset");

	profile_set.Load("uniprot_sprot_201610.profileset");
	profile_set.Save("tmp.profile_set");

	profile_set.Print("uniprot_sprot_201610.txt");
	return;

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

}

void CAFA2() {
	GOTermSet go_set;
	go_set.ParseGo("../data/gene_ontology_edit.obo.2014-01-01");
	go_set.Save("go_140101.gotermset");

	ProteinSet train_set;
	train_set.ParseRawTxt(go_set, "../data/201401/Seq", "../data/201401/Ann");
	train_set.Save("cafa2_train_170307.proteinset");

	ProteinSet test_set;
	test_set.ParseRawTxt(go_set, "../data/CAFA2/Seq", "../data/CAFA2/Ann");
	test_set.Save("cafa2_test_170307.proteinset");
}

int main() {

	CAFA2();
	system("pause");
    return 0;
}
