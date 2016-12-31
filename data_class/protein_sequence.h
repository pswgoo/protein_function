#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>

enum LogStatus {SILENT, PART_LOG, FULL_LOG};

//enum SubtractProteinMode {MODIFIED_GO_TERM, ADD_NEW_GO_TERM, COMPLETE_NEW_ANNOTATION};

/**
 * @enumerator enum all amino types
 */
enum AminoType {NON, ALA, ARG, ASN, ASP, CYS, GLU, GLN, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL};
const std::string kTypeString[] = { "NON", "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"};
const char kTypeChar[]          = { '0',    'A',   'R',   'N',   'D',   'C',   'E',   'Q',   'G',   'H',   'I',   'L',   'K',   'M',   'F',   'P',   'S',   'T',   'W',   'Y',   'V'  };

AminoType GetAminoType(const std::string &x);

inline std::string Get3LetterAminoName(AminoType type) {
	return kTypeString[type];
}

inline char Get1LetterAminoName(AminoType type) {
	return kTypeChar[type];
}

enum GoType {MF, BP, CC, GO_TYPE_SIZE};
const std::string kGoTypeStr[] = { "molecular_function", "biological_process", "cellular_component" };

struct Protein {

	const std::vector<int>& go_term(GoType go_type) const {
		return go_terms_[go_type];
	}

	friend boost::serialization::access;
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & id_  & sequence_ & go_terms_;
	}

	Protein() = default;
	Protein(const std::string& id, const std::string& sequence) : id_(id), sequence_(sequence) {
		go_terms_.resize(GoType::GO_TYPE_SIZE);
	}

	std::string id_;
	std::string sequence_;
	std::vector<std::vector<int>> go_terms_;
};

class ProteinSet {

public:
	std::vector<Protein> proteins() { return proteins_; }

	bool Has(const std::string& id) const { return protein_indices_.count(id) > 0; }

	int Size() const { return (int)proteins_.size(); }

	const Protein& operator[](const std::string& id) const { return proteins_[protein_indices_.at(id)]; }
	const Protein& operator[](int idx) const { return proteins_[idx]; }

	int ParseRawTxt(const std::string& sequence_file, const std::string& mf_go_file, const std::string& bp_go_file, const std::string& cc_go_file, bool only_left_indexed = true);

	void Save(const std::string& file_name) const;

	int Load(const std::string& file_name);

	void set_proteins(const std::vector<Protein>& proteins) {
		proteins_.clear();
		protein_indices_.clear();
		proteins_ = proteins;
		for (int i = 0; i < proteins_.size(); ++i)
			protein_indices_[proteins_[i].id_] = i;
	}

private:
	friend boost::serialization::access;
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & proteins_  & protein_indices_;
	}

private:
	std::vector<Protein> proteins_;
	std::unordered_map<std::string, int> protein_indices_;
};

inline GoType GoTypeStrToTypeId(std::string type_str) {
	for (int i = MF; i < GO_TYPE_SIZE; ++i)
		if (kGoTypeStr[i] == type_str)
			return (GoType)i;
	return GO_TYPE_SIZE;
}
