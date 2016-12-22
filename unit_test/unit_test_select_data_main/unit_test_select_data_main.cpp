#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <map>

#include "data_class/protein_sequence.h"

using namespace std;

std::unordered_set<std::string> Exp = {"EXP","IDA","IPI","IMP","IGI","IEP","TAS","IC"};
std::unordered_set<std::string> Spc_cafa1 = {"BACSU","PSEAE","STRPN","ARATH","YEAST","XENLA","HUMAN","MOUSE","RAT","DICDI","ECOLI"};

void OutputStatistic(const ProteinSequenceSet& protein_set, string file_name) {
	map<char, int> go_type_code = {{'C', 1}, {'F', 2}, {'P', 3}};
	map<string, int> spicies_cnt;
	map<string, vector<int>> go_type_cnt;
	for (const ProteinSequence& protein : protein_set.protein_sequences()) {
		spicies_cnt[protein.species()]++;
		vector<bool> has(4);
		for (ProteinSequence::GOType go : protein.go_terms())
			has[go_type_code[go.term_[0]]] = true;
		if (go_type_cnt.count(protein.species()) == 0)
			go_type_cnt[protein.species()].resize(4);
		for (size_t i = 0; i < has.size(); ++i)
			if (has[i])
				go_type_cnt[protein.species()][i]++;
	}
	
	ofstream fout(file_name);
	fout << "spice, count, C, F, P" << endl;
	for (auto it = spicies_cnt.begin(); it != spicies_cnt.end(); ++it) {
		fout << it->first << ", " << it->second << ", " << go_type_cnt[it->first][1] << ", " << go_type_cnt[it->first][2] << ", " << go_type_cnt[it->first][3] << endl;
	}
	fout.close();
}

int main() {
	ProteinSequenceSet u201101set;
	ProteinSequenceSet goa201101set;
	ProteinSequenceSet u201112set;
	ProteinSequenceSet u201101_allset;
	ProteinSequenceSet goa201101_hasaccession_set;
	u201101_allset.Load("uniprot_201101.proteinset");
	u201101set.Load("uniprot_201101_filtered_indexed.proteinset");
//	u201101set.SaveToFasta("uniprot_201101_filtered_indexed.fasta");
//	u201101set.Load("cafa1_select_set_0.proteinset");
//	u201101set.SaveToFasta("cafa1_select_set_0.fasta");
//	return 0;
	goa201101set.Load("goa_201101_filered_indexed.proteinset");
	goa201101_hasaccession_set.Load("goa_201101_indexed_filterbyevidence.proteinset");
	goa201101_hasaccession_set.SaveToString("has_accession_tmp.txt");
	
	u201112set.Load("uniprot_sprot_201112.proteinset");
	u201112set.FilterProteinSequenceBySpicies(Spc_cafa1);
	u201112set.FilterGoByEvidence(Exp);
	u201112set.FilterNotIndexedProtein();
	clog << "Uniprot 201112 after filtered, there are " << u201112set.protein_sequences().size() << " protein left" << endl;
	
	u201112set.FilterGoCc();
	u201112set.FilterNotIndexedProtein();
	u201101set.FilterGoCc();
	u201101set.FilterNotIndexedProtein();
	goa201101set.FilterGoCc();
	goa201101set.FilterNotIndexedProtein();
	goa201101_hasaccession_set.FilterGoCc();
	goa201101_hasaccession_set.FilterNotIndexedProtein();
	clog << "after filter cc, u201112set.size = " << u201112set.protein_sequences().size() << ", u201101set.size = " << u201101set.protein_sequences().size()
		<< ", goa201101set.size = " << goa201101set.protein_sequences().size() << ", goa201101_hasaccession_set.size = " << goa201101_hasaccession_set.protein_sequences().size() << endl;
	
	ProteinSequenceSet select_set;
	select_set = u201112set.SubtractByNameSpice(u201101set);
	clog << "u201112set - u201101set, left " << select_set.protein_sequences().size() << " protein" << endl;
	select_set = select_set.SubtractByNameSpice(goa201101set);
	clog << "u201112set - u201101set - goa201101set, left " << select_set.protein_sequences().size() << " protein" << endl;
	select_set = select_set.SubtractByAccession(goa201101_hasaccession_set);
	clog << "u201112set - u201101set - goa201101set - goa201101_hasaccession_set, left " << select_set.protein_sequences().size() << " protein" << endl;
	select_set.ReserveByNameSpice(u201101_allset);
	select_set.Save("cafa1_select_set_0.proteinset");
	clog << "Finally, left " << select_set.protein_sequences().size() << " protein" << endl;
	select_set.SaveToString("cafa1_select_set_0.txt");
	OutputStatistic(select_set, "cafa1_filter_cc.csv");
	return 0;
}
