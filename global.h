#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <tuple>
#include <queue>
#include <map>
#include <regex>
using namespace std;

enum NODETYPE {
	SELECT, PROJECT, JOIN, UNION, TABLE, FRAGMENT
};    // 查询树中的节点类型

enum RELATION {
	EQ, NE, G, GE, L, LE
};    // 谓词重的关系类型  =, !=, >, >=, <, <=

enum PREDTYPE {
	INT, CHAR, TAB
};    // 谓词类型

string ntype_to_token(NODETYPE ntype){
    switch (ntype){
        case SELECT: return "select"; case PROJECT: return "project";
        case JOIN: return "join"; case UNION: return "union";
        case TABLE: return "table"; case FRAGMENT: return "fragment";
    }
}     

string rela_to_token(RELATION rel){
    switch (rel){
        case EQ: return "="; case L: return "<"; case G: return ">";
        case LE: return "<="; case GE: return ">="; case NE: return "!=";
    }
}

struct predicateV{
	string table_name;
	string attribute;
	RELATION rel;
	int value;
	string get_str(){
		return table_name+"."+attribute+rela_to_token(rel)+to_string(value);
	}
};     // INT 型谓词 e.g Table.attribute = 12

struct predicateS{
	string table_name;
	string attribute;
	RELATION rel;
	string value;
	string get_str(){
		return table_name+"."+attribute+rela_to_token(rel)+value;
	}
};      // CHAR 型谓词  e.g Table.attribute = 'string'

struct predicateT{
	string left_table;
	string left_attr;
	RELATION rel;
	string right_attr;
	string right_table;
	string get_str(){
		return left_table+"."+left_attr+rela_to_token(rel)+right_table+"."+right_attr;
	}
};     // TAB 型谓词 e.g Table1.attribute = Table2.attribute

vector<string> split(string str, string delim) {
    regex re{ delim };
    return vector<std::string> {
        sregex_token_iterator(str.begin(), str.end(), re, -1),
            sregex_token_iterator()
    };
}     // 分词

int c_in_str(string str, const char c){
    int count = 0;
    for(int i = 0;i < str.size();i ++)
        if(str[i] == c)
            count ++;
    return count;
} // 统计字符 c 在字符串 str 中出现的次数

void trim(string &s)
{
    int index = 0;
    if(!s.empty())
        while( (index = s.find(' ',index)) != string::npos)
            s.erase(index,1);
}  //去掉字符串 s 中包含的空格
