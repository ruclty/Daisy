#include <mysql.h>
#include <string>
using namespace std;

string get_sql();
string excute_sql(string excute_sql);
void update_global_dict(int frag_id, string table_name);

void create_temp_table();
void sort_table();


class MySql{
	private:
		MYSQL mysql;
		void Init(MYSQL *mysql);
		void connection(MYSQL *mysql, const char *host, const char* user, const char *passwd, const char* db, unsigned int port);
	public:
		void do_loocal_Query(Operator sql);
		void get_results()
}