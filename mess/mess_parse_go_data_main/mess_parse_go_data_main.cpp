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
std::unordered_set<std::string> kExp = {"EXP","IDA","IPI","IMP","IGI","IEP","TAS","IC"};
const vector<string> kLeftSpecies = {"BACSU","PSEAE","STRPN","ARATH","YEAST","XENLA","HUMAN","MOUSE","RAT","DICDI","ECOLI"};

struct Association {
    int id_;
    int term_id_;
    int gene_product_id_;
};

struct Term {
    int term_id_;
    string name_;
    string type_;
    string acc_;
};

vector<Association> association_table;
unordered_map<int, string> assc_to_evidence_code;
unordered_map<int, Term> go_term_table;
unordered_map<int, string> gene_product_synonym;

vector<string> SplitString(const string &str, string sep = "\t") {
    vector<string> ret_str;
    boost::split(ret_str, str, boost::is_any_of(sep));
    return ret_str;
}

bool IsLeftSpecies(string str) {
    bool is_name = false;
    for (const string& spice : kLeftSpecies)
        if (str.find("_" + spice) != string::npos) {
            is_name = true;
            break;
        }
    return is_name;
}

void Init() {
    const string kAssociationFile = "/home/zhangyis13/protein/InFile/GO/go_201101-assocdb-tables/association.txt";
    const string kTermFile = "/home/zhangyis13/protein/InFile/GO/go_201101-assocdb-tables/term.txt";
    const string kEvidenceFile = "/home/zhangyis13/protein/InFile/GO/go_201101-assocdb-tables/evidence.txt";
    const string kSynonymFile = "/home/zhangyis13/protein/InFile/GO/go_201101-assocdb-tables/gene_product_synonym.txt";

    string line;
    ifstream fin1(kAssociationFile);
	if (fin1.is_open())
		clog << "Open assoc file OK" << endl;
	int cnt = 0;
    while (getline(fin1,line)) {
		++cnt;
        vector<string> strs = SplitString(line);
		if (strs.size() < 3)
			clog << "assoc file str size < 3: it is " << cnt << " line: " << line << endl;
		else {
			Association asso = {atoi(strs[0].data()), atoi(strs[1].data()), atoi(strs[2].data())};
			association_table.push_back(asso);
		}
    }
    clog << "Load " << association_table.size() << " association instance " << endl;

    ifstream fin2(kTermFile);
    while (getline(fin2,line)) {
        vector<string> strs = SplitString(line);
		if (strs.size() < 4)
			clog << "term file str size < 4: it is " << line << endl;
        Term term = {atoi(strs[0].data()), strs[1], strs[2], strs[3]};
        if (term.type_ == "molecular_function") {
            term.type_ = "F";
            go_term_table[term.term_id_] = term;
        }
        else if (term.type_ == "biological_process") {
            term.type_ = "P";
            go_term_table[term.term_id_] = term;
        }
        else if (term.type_ == "cellular_component") {
            term.type_ = "C";
            go_term_table[term.term_id_] = term;
        }
    }
    clog << "Load " << go_term_table.size() << " go term instance " << endl;

    ifstream fin3(kEvidenceFile);
    while (getline(fin3, line)) {
        vector<string> strs = SplitString(line);
		if (strs.size() < 3)
			clog << "evidence file str size < 3: it is " << line << endl;
        assc_to_evidence_code[atoi(strs[2].data())] = strs[1];
    }
    clog << "Load " << assc_to_evidence_code.size() << " evidence instance " << endl;

    ifstream fin4(kSynonymFile);
    while (getline(fin4, line)) {
        vector<string> strs = SplitString(line);
		if (strs.size() < 2)
			clog << "synonym file str size < 2: it is " << line << endl;
		else
			gene_product_synonym[atoi(strs[0].data())] = strs[1];
    }
    clog << "Load " << gene_product_synonym.size() << " gene_product_synonym instance " << endl;
}

ProteinSequenceSet GetProteinSequenceSet() {
//    sort(association_table.begin(), association_table.end(), [](const Association& a1, const Association& a2) { return a1.gene_product_id_ < a2.gene_product_id_; });
    ProteinSequenceSet protein_set;
    for (size_t i = 0; i < association_table.size(); ) {
        int cur_protein_id = association_table[i].gene_product_id_;
        if (gene_product_synonym.count(cur_protein_id) == 0 || !IsLeftSpecies(gene_product_synonym[cur_protein_id])) {
			++i;
            continue;
		}
        ProteinSequence protein;
        vector<ProteinSequence::GOType> go_terms;
        while (i < association_table.size() && association_table[i].gene_product_id_ == cur_protein_id) {
            string evidence;
            if (assc_to_evidence_code.count(association_table[i].id_) > 0)
                evidence = assc_to_evidence_code[association_table[i].id_];
            if (evidence.size() > 3)
                clog << "Warning: Evidence: " << evidence << endl;
            if (kExp.count(evidence) > 0 && go_term_table.count(association_table[i].term_id_) > 0) {
                int go_id;
                Term term = go_term_table[association_table[i].term_id_];
                if (term.acc_.find(":") != string::npos)
                    go_id = atoi(term.acc_.substr(term.acc_.find(":")+1).data());

                go_terms.push_back({go_id, term.type_ + ": " + term.name_, evidence});
            }
			++i;
        }
        string name_spice = gene_product_synonym[cur_protein_id];
        vector<string> tmp_strs = SplitString(name_spice, "_");
        protein.set_go_terms(go_terms);
        protein.set_name(tmp_strs[0]);
        protein.set_species(tmp_strs[1]);
        protein_set.add_protein_sequence(protein);
        if (protein_set.protein_sequences().size() % 100 == 0)
            clog << "\r" << protein_set.protein_sequences().size() << " loaded";
    }
    clog << endl;
    clog << "Total load " << protein_set.protein_sequences().size() << " protein instances" << endl;
    return protein_set;
}

void PrintProtein(const ProteinSequenceSet& protein_set, string file_name) {
    vector<ProteinSequence> vec_proteins = protein_set.protein_sequences();
    sort(vec_proteins.begin(), vec_proteins.end(), [](const ProteinSequence &p1, const ProteinSequence &p2) {
        return p1.name() + "_" + p1.species() < p2.name() + "_" + p2.species();
    });
    ofstream fout(file_name);
    for (ProteinSequence protein : vec_proteins) {
    vector<ProteinSequence::GOType> goterms = protein.go_terms();
        sort(goterms.begin(), goterms.end(), [](const ProteinSequence::GOType &term1, const ProteinSequence::GOType& term2) {return term1.id_ < term2.id_;});
        for (size_t i = 0; i < goterms.size(); ++i) {
            fout << protein.name() + "_" + protein.species() + "\t" + "GO:" << setw(7) << setfill('0') << goterms[i].id_ << endl;
        }
    }
}

int main() {
    //Init();

	GOTermSet go_term_set;
	go_term_set.ParseGo("C:/psw/cafa/CAFA3/Ontology/gene_ontology_edit.obo.2016-06-01");
	clog << "Total load " << go_term_set.go_terms().size() << " GOs";

	//ProteinSequenceSet cafa3_set;
	//cafa3_set.Load("cafa1_select_set_0.proteinset");
	//PrintProtein(cafa3_set, "cafa1_0_align.txt");
	//return 0;
 //   //ProteinSequenceSet go_protein_set = GetProteinSequenceSet();
 //   //go_protein_set.Save("go_201101.proteinset");
 //   //go_protein_set.SaveToString("go_201101.txt");
	//ProteinSequenceSet go_protein_set2(FULL_LOG);
	//go_protein_set2.Load("go_201101.proteinset");
	//go_protein_set2.FilterGoCc();
	//go_protein_set2.FilterNotIndexedProtein();
	//clog << "After filter, left " << go_protein_set2.protein_sequences().size() << " instance " << endl;
	//ProteinSequenceSet cafa1_set;
	//cafa1_set.Load("cafa1_select_set_0.proteinset");
	//ProteinSequenceSet cafa2_set;
	//cafa2_set = cafa1_set.SubtractByNameSpice(go_protein_set2);
	//clog << "Left " << cafa2_set.protein_sequences().size() << " instance " << endl;
	//cafa2_set.Save("cafa1_select_set_1.proteinset");
	//cafa2_set.SaveToString("cafa1_select_set_1.txt");
	system("pause");
    return 0;
}
