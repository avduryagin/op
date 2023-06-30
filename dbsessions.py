import psycopg2 as ps
import pandas as pd
import os

class Session:
    def __init__(self,host= "vps-pgsql01.asuproject.ru",
                 dbname = "oisp_160_dev",user = "ufam_oisp_160",
                 password = "ufam_oisp_160",port = 5432):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port
        self.session=None
        self.path=os.path.join(os.getcwd(),"sql")
        self.sql=""
        self.auth="select PSBS_SECURE.SetContext('root', cast (NOW() as TIMESTAMP), cast (NOW() as TIMESTAMP), '6766', 67)"


    def open(self):
        self.session=ps.connect(host=self.host,dbname=self.dbname,
                                user=self.user,password=self.password)
        auth=pd.read_sql(self.auth,self.session)
    def wrap(self,tup=tuple()):
        if tup is None:
            return tup
        assert isinstance(tup,tuple), "an argument should be a tuple"
        if len(tup)==1:
            if str==type(tup[0]):
                return "('{item}')".format(item=tup[0])
            else:
                return '({item})'.format(item=tup[0])
        elif len(tup)==0:
            return "()"
        else:
            return tup

    def get_activities(self,plan_id=343043,field=(1200186503,),shop=(323938109,),enterprise=(1200186103,),sdate='01.01.2022',
                      edate='01.01.2023',file='activities.sql'):
        def get_objects(enterprise=None,shop=None,field=None, ):
            di = dict({"ENTERPRISE": enterprise, "SHOP": shop,"FIELD": field})
            #print(di)
            default = "(SELECT o.activity_id from sf_activity_own o)"
            string = "(SELECT o.activity_id from sf_activity_own o WHERE o.entity_id in {value} AND o.entity_name = '{key}')"
            sql = ""

            i = 0
            joint = False
            for k in di:
                v = di[k]
                if v is not None:
                    string_ = string.format(key=k, value=v)
                    joint = True
                    if i > 0:
                        string_ = " intersect " + string_
                    sql += string_
                    i += 1
            if not joint:
                return default
            return sql

        with open(os.path.join(self.path,file),"r",encoding='utf8') as f:
            sql=str(f.read())
        f.close()

        objects=get_objects(enterprise=self.wrap(enterprise),shop=self.wrap(shop),field=self.wrap(field))
        sql=sql.format(plan_id=plan_id,sdate=sdate, edate=edate,objects=objects)
        self.sql = sql
        return pd.read_sql(sql,self.session)
    def get_wells(self,wells_id=(),Field=25010255385151,file='wells.sql'):
        with open(os.path.join(self.path,file),"r",encoding='utf8') as f:
            sql=str(f.read())
        f.close()
        sql=sql.format(Field, wells_id)
        return pd.read_sql(sql,self.session)

    def update(self,data=pd.DataFrame(columns=['id','executor','exec_begin','exec_end']),plan_id=343043,file='update.sql',record=False):
        missed=[]

        mask=~data['executor'].isnull()
        cursor=self.session.cursor()
        for index in data.loc[mask].index:
            executor=data.at[index,'executor']
            activity=data.at[index,'id']
            sdate=data.at[index,'exec_begin']
            edate = data.at[index, 'exec_end']
            with open(os.path.join(self.path, file), "r", encoding='utf8') as f:
                sql = str(f.read())
            f.close()
            sql = sql.format(id=activity, execid=executor,sdate=sdate,edate=edate,plan=plan_id,distributed=1)
            #self.sql=sql
            cursor.execute(sql)
            if (cursor.statusmessage!='UPDATE 1')& record:
                missed.append(activity)
        self.session.commit()
        return True

    def reset(self, data=(), plan_id=343043,
               file='reset.sql'):

        cursor = self.session.cursor()
        for a in data:
            executor = "null"
            activity = a
            with open(os.path.join(self.path, file), "r", encoding='utf8') as f:
                sql = str(f.read())
            f.close()
            sql = sql.format(id=activity, execid=executor,  plan=plan_id, distributed=0)

            cursor.execute(sql)
        self.session.commit()
        self.sql = sql
        return True

        #return pd.read_sql(sql,self.session)

    def get_roads(self,wells_id=(),file='road_map.sql'):
        with open(os.path.join(self.path,file),"r",encoding='utf8') as f:
            sql=str(f.read())
        f.close()
        f.close()
        sql=sql.format(wells_id)
        return pd.read_sql(sql,self.session)
    def get_executors(self,plan_id=343043,sdate='01.01.2022',types=(),shops=(),fields=(),well_pads=(),wells=(),objtypes=(),activities=(),file='executors.sql'):
        with open(os.path.join(self.path,file),"r",encoding='utf8') as f:
            sql=str(f.read())
        f.close()
        sql=sql.format(plan_id=plan_id,sdate=sdate,types=self.wrap(types),shops=self.wrap(shops),fields=self.wrap(fields),well_pads=self.wrap(well_pads),
                       wells=self.wrap(wells),objtypes=self.wrap(objtypes),activities=self.wrap(activities))
        return pd.read_sql(sql,self.session)

    def get_executors_v1(self,plan_id=343043,sdate='01.01.2022',types=(),shops=(),fields=(),well_pads=(),wells=(),enterprises=(),file='executors_obj.sql'):
        with open(os.path.join(self.path,file),"r",encoding='utf8') as f:
            sql=str(f.read())
        f.close()
        sql=sql.format(sdate=sdate,types=self.wrap(types),shops=self.wrap(shops),fields=self.wrap(fields),well_pads=self.wrap(well_pads),
                       wells=self.wrap(wells),enterprises=self.wrap(enterprises))
        self.sql=sql
        return pd.read_sql(sql,self.session)
    def get_joint_by_rules(self,types_id=(),rule_id=1247804,file='compatibility_joines.sql'):
        with open(os.path.join(self.path,file),"r",encoding='utf8') as f:
            sql=str(f.read())
        f.close()
        sql=sql.format(rule=rule_id, type_id_tuple=types_id)
        return pd.read_sql(sql,self.session)



