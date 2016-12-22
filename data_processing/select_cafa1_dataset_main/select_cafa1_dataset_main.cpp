#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "data_class/protein_sequence.h"

using namespace std;

std::unordered_set<std::string> Exp = {"EXP","IDA","IPI","IMP","IGI","IEP","TAS","IC"};
std::unordered_set<std::string> Spc_cafa1 = {"BACSU","PSEAE","STRPN","ARATH","YEAST","XENLA","HUMAN","MOUSE","RAT","DICDI","ECOLI"};

void SaveUniprot201101Set() {
	ProteinSequenceSet u201101set(20110100, LogStatus::FULL_LOG);
	u201101set.ParseUniprotXml("/protein/raw_data/uniprot_sprot_201101.xml");
	u201101set.Save("/protein/work/uniprot_201101.proteinset");
	u201101set.FilterProteinSequenceBySpicies(Spc_cafa1);
	u201101set.FilterGoByEvidence(Exp);
	u201101set.FilterNotIndexedProtein();
	u201101set.Save("/protein/work/uniprot_201101_filtered_indexed.proteinset");
}

void SaveCafa1AllSpeciesTrainSet() {
	ProteinSequenceSet u201101set(20110100, LogStatus::FULL_LOG);
	u201101set.Load("/protein/work/uniprot_201101.proteinset");
	u201101set.FilterGoByEvidence(Exp);
	u201101set.FilterGoCc();
	u201101set.FilterNotIndexedProtein();
	clog << "After filter, left " << u201101set.protein_sequences().size() << " proteins " << endl;
	u201101set.Save("/protein/work/cafa1_all_species_trainset.proteinset");
	u201101set.SaveToFasta("/protein/work/cafa1_all_species_trainset.fasta");
}

void SaveCafa1OnlySpeciesTrainSet() {
	ProteinSequenceSet u201101set(20110100, LogStatus::FULL_LOG);
	u201101set.Load("/protein/work/uniprot_201101.proteinset");
	u201101set.FilterProteinSequenceBySpicies(Spc_cafa1);
	u201101set.FilterGoByEvidence(Exp);
	u201101set.FilterGoCc();
	u201101set.FilterNotIndexedProtein();
	clog << "After filter, left " << u201101set.protein_sequences().size() << " proteins " << endl;
	u201101set.Save("/protein/work/cafa1_only_species_trainset.proteinset");
	u201101set.SaveToFasta("/protein/work/cafa1_only_species_trainset.fasta");
}

void SaveCafa1TestSet() {
	ProteinSequenceSet cafa1_set;
	cafa1_set.Load("/protein/work/cafa1_select_set_0.proteinset");
	unordered_set<string> list;
	ifstream fin("ProteinList.in");
	string line;
	while(fin >> line)
		list.insert(line);
	vector<ProteinSequence> left_proteins;
	for (const auto& protein : cafa1_set.protein_sequences())
		if (list.count(protein.name() + "_" + protein.species()) > 0)
			left_proteins.push_back(protein);
	cafa1_set.set_protein_sequences(left_proteins);
	cafa1_set.Save("/protein/work/cafa1_testset.proteinset");
	cafa1_set.SaveToFasta("/protein/work/cafa1_testset.fasta");
	cafa1_set.SaveToString("/protein/work/baseline/cafa1_testset.txt");
}

void SaveCafa1TestSet(const string& db_file, const string& list_file, const string& output_file_head) {
	ProteinSequenceSet u201101set(20110100, LogStatus::FULL_LOG);
	u201101set.Load(db_file);
	u201101set.BuildNameIndex();
	unordered_map<string, ProteinSequence> protein_map;
	ifstream fin(list_file);
	string name, go;
	while (fin >> name >> go) {
		ProteinSequence &protein = protein_map[name];
		protein.set_name(name.substr(0, name.find("_")));
		protein.set_species(name.substr(name.find("_") + 1));
		if (u201101set.HasProtein(name))
			protein.set_sequence(u201101set.QueryByName(name).sequence());
		else {
			clog << "Error: " << name << " cannot find sequence in database" << endl;
			continue;
		}
		int go_id = atoi(go.substr(go.find(":")+1).c_str());
		protein.add_go_term({go_id, "", ""});
	}
	clog << "Load " << protein_map.size() << " proteins" << endl;
	ProteinSequenceSet test_set;
	for (const auto& pr : protein_map)
		if (!pr.second.sequence().empty())
			test_set.add_protein_sequence(pr.second);
	clog << "Begin save " << test_set.protein_sequences().size() << " proteins" << endl;
	test_set.Save(output_file_head + ".proteinset");
	test_set.SaveToFasta(output_file_head + ".fasta");
	test_set.SaveToString(output_file_head + ".txt");
}

int main() {
//	SaveCafa1AllSpeciesTrainSet();
//	SaveCafa1OnlySpeciesTrainSet();
	SaveCafa1TestSet("/protein/work/uniprot_201101.proteinset", "cafa1_testset_bp_list.txt", "cafa1_testset_bp");
	SaveCafa1TestSet("/protein/work/uniprot_201101.proteinset", "cafa1_testset_mf_list.txt", "cafa1_testset_mf");
	
	clog << "Complete" << endl;
	return 0;
}
