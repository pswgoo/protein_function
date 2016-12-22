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
	u201101set.Save("uniprot_201101.proteinset");
	u201101set.FilterProteinSequenceBySpicies(Spc_cafa1);
	u201101set.FilterGoByEvidence(Exp);
	u201101set.FilterNotIndexedProtein();
	u201101set.Save("uniprot_201101_filtered_indexed.proteinset");
}

ProteinSequenceSet GetHasAccessionProteinSet(const ProteinSequenceSet &protein_set) {
	ProteinSequenceSet ret_protein_set;
	for (const ProteinSequence& protein : protein_set.protein_sequences())
		if (!protein.accessions().empty() && protein.species().empty()){
			ret_protein_set.add_protein_sequence(protein);
		}
	return ret_protein_set;
}

int main() {
	//SaveUniprot201101Set();
	//return 0;
	ProteinSequenceSet protein_set(20110100, LogStatus::FULL_LOG);
	protein_set.ParseGoa("/protein/raw_data/gene_association.goa_uniprot.91");
	protein_set.Save("goa_201101.proteinset");
	//protein_set.SaveToString("goa_tmp.txt");
	
	ProteinSequenceSet has_accession_set = GetHasAccessionProteinSet(protein_set);
	clog << "has_accession_set.size = " << has_accession_set.protein_sequences().size() << endl;
	has_accession_set.FilterGoByEvidence(Exp);
	has_accession_set.FilterNotIndexedProtein();
	clog << "has_accession_set indexed size = " << has_accession_set.protein_sequences().size() << endl;
	has_accession_set.Save("goa_201101_indexed_filterbyevidence.proteinset");
	
	protein_set.FilterProteinSequenceBySpicies(Spc_cafa1);
	protein_set.FilterGoByEvidence(Exp);
	//protein_set.FilterGoCc();
	
	clog << "After filter, there " << protein_set.protein_sequences().size() << " sequence left" << endl;
	
	ProteinSequenceSet protein_indexed_set = protein_set;
	protein_indexed_set.FilterNotIndexedProtein();
	protein_indexed_set.Save("goa_201101_filered_indexed.proteinset");
	protein_indexed_set.SaveToString("goa_91_indexed.txt");
	clog<<"GO !=  0 num = "<< protein_indexed_set.protein_sequences().size() <<endl;
	clog<<"GO != 0  &&  GO=0 all  num = "<<protein_set.protein_sequences().size() <<endl;
	clog << "Complete" << endl;
}
