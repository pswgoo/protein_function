#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "data_class/protein_sequence.h"

using namespace std;

int main() {

	ProteinSequenceSet protein_set(20150100, LogStatus::FULL_LOG);
	size_t suc_cnt = 0;
	if ((suc_cnt = protein_set.ParseUniprotXml("/protein/raw_data/uniprot_sprot_201501.xml")) > 0) {
	  cout << suc_cnt << " success" << endl;
	  protein_set.Save("uniprot_201501.protein_set");
	}
	else
		cerr << "Failed" << endl;
	
	protein_set.Load("uniprot_201501.protein_set");
	ProteinSequence seq =  protein_set.protein_sequences().at(0);
	std::vector<AminoType> ami = seq.sequence();
	for(size_t i = 0; i < ami.size(); i++) {
	  cout<<Get3LetterAminoName(ami[i]);
	}
	
	
	
/*fout top 100 protein name species sequence go */	
	/*
	ofstream fout;
	fout.open("top100.txt");
	int cnt = 0;
	std::vector<ProteinSequence> seq_top = protein_set.protein_sequences();
	for(auto it = seq_top.begin(); it != seq_top.end(); ++it) {
	  	++cnt;
	  	if(cnt > 100) break;
	  	fout<<it->name()<<endl;
		fout<<it->species()<<endl;
	  
	  	std::vector<AminoType> ami_top = it->sequence();
		for(size_t i = 0; i < ami.size(); i++) {
	  		fout<<Get3LetterAminoName(ami[i]);
	  	}
	  	fout<<endl;
	  	
	  	std::vector<ProteinSequence::GOType> go_top = it->go_terms();
	  	for(std::vector<ProteinSequence::GOType>::iterator it2 = go_top.begin(); it2 != go_top.end(); ++it2){
	  		fout<<it2->id_<<"	"<<(*it2).term_<<"	"<<(*it2).evidence_<<"	";
	 	 }
	 	 fout<<endl;
	}
	fout.close();
	*/
	protein_set.SaveToFasta("20150100fasta.txt");
	/*
	ProteinSequenceSet protein_set(20150100, LogStatus::FULL_LOG);
	protein_set.ParseGoa("/home/kochiyaocean/protein_function/test.txt");
	protein_set.Save("goa.protein_set");	
	protein_set.Load("goa.protein_set");
	for(size_t i = 0; i < protein_set.protein_sequences().size(); i++) {
	  cout<<protein_set.protein_sequences().at(i).ToString();
	}
	ProteinSequence seq =  protein_set.protein_sequences().at(0);
	std::vector<AminoType> ami = seq.sequence();
	for(size_t i = 0; i < ami.size(); i++) {
	  cout<<Get3LetterAminoName(ami[i]);
	}
	clog << "Complete" << endl;
	*/
	return 0;
}
