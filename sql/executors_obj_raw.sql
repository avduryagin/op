WITH contractdata
     AS (SELECT c.id       contract,
                m.class_id meropType,
                c.executor,
                c.start_date,
                c.end_date
         FROM   sf_contract c,
                sf_contract_cost m
         WHERE  C.id = m.contract_id
                AND c.end_date > cast(to_date('{sdate}','dd.mm.yyyy') as date)),
     contractstoshop
     AS (SELECT *
         FROM   sf_contract_shop scs
                inner join contractdata
                        ON scs.contract_id = contractdata.contract
         WHERE (shop_id is null) or shop_id in {shops}),
     executorsbyorg
     AS (SELECT c.*,
                wc.id                                 WORK_CALENDAR_ID,
                wt.id                                 WORK_TIME_ID,
                fp.id                                 SPECIFICATION_FPLAN_ID,
                mp.id                                 MOBILITY_TYPE_ID,
                rbp.code                              RETURN_TO_BASE_CODE,
                wsch.is_multi_task,
                wsch.is_shift_work,
                wsch.overtime_day,
                wsch.overtime_month,
                wsch.is_minz,
                wsch.max_stop_hour,
                exloc.facility_s                      BASE_OBJ_ID,
                exloc.facility_t                      BASE_OBJ_TYPE,
                Coalesce(wsch.is_extra_contractor, 0) IS_EXTRA_CONTRACTOR,
                c.is_hidden                           IS_DELETED
         FROM   sf_class c
                left join (SELECT w.*,
                                  l.cf
                           FROM   sf_class w,
                                  sf_claslink l
                           WHERE  l.ct = w.id
                                  AND w.sf_ref_type = 'WORK_CALENDAR') wc
                       ON c.id = wc.cf
                left join (SELECT w.*,
                                  l.cf
                           FROM   sf_class w,
                                  sf_claslink l
                           WHERE  l.ct = w.id
                                  AND w.sf_ref_type = 'WORK_TIME') wt
                       ON c.id = wt.cf
                left join (SELECT w.*,
                                  l.cf
                           FROM   sf_class w,
                                  sf_claslink l
                           WHERE  l.ct = w.id
                                  AND w.sf_ref_type = 'F_PLAN') fp
                       ON c.id = fp.cf
                left join (SELECT w.*,
                                  l.cf
                           FROM   sf_class w,
                                  sf_claslink l
                           WHERE  l.ct = w.id
                                  AND w.sf_ref_type = 'MOBILITY') mp
                       ON c.id = mp.cf
                left join (SELECT w.*,
                                  l.cf
                           FROM   sf_class w,
                                  sf_claslink l
                           WHERE  l.ct = w.id
                                  AND w.sf_ref_type = 'EQ_RETURN') rbp
                       ON c.id = rbp.cf
                left join sf_work_schedule wsch
                       ON c.id = wsch.contractor_id
                left join sf_work_schedule parent_wsch
                       ON c.parent_id = parent_wsch.contractor_id
                left join if_executor_location exloc
                       ON c.id = exloc.if_executor_s
         WHERE  c.sf_ref_type IN ( 'BR' )
                AND c.is_hidden = 0
                AND ( c.parent_id IN (SELECT contractstoshop.executor
                                      FROM   contractstoshop) )),
                                      
     executor
     AS (SELECT c.id,
                c.sf_ref_type,
                Coalesce (c2.name, c.name)      AS ORGNAME,
                Coalesce (c2.name, c.name, ' ') AS NAME,
                c.code,
                c.parent_id
         FROM   sf_class c
                inner join sf_metalink m
                        ON m.cf = 'EXECUTOR'
                           AND m.ct = c.sf_ref_type
                left join sf_class c2
                       ON c.parent_id = c2.id
                       ),
          merop
     AS (SELECT c.*
         FROM   sf_class c,
                sf_metalink m
         WHERE  c.sf_ref_type = m.ct
                AND m.cf = 'MEROP'),
                
      objects as 
      
      (SELECT e.*
      FROM SF_EXECUTOR_OBJECT  e
      INNER JOIN sf_class c
           ON c.ID = e.EXECUTOR_ID AND c.SF_REF_TYPE IN ('PR', 'BR') where (e.well_id is null or (e.well_id is not null and e.well_id in {wells})) or (e.well_pad_id is null or (e.well_pad_id is not null and e.well_pad_id in {well_pads})) or 
((e.field_id is null) or (e.field_id is not null and e.field_id in {fields})) or ((e.shop_id isnull) or (e.shop_id is not null and e.shop_id in {shops})) or ((e.enterprise_id is null) or (e.enterprise_id is not null and e.shop_id in {enterprises}))), 
    
 executors as (SELECT meropContract.type_id,
       executorsbyorg.* from executorsbyorg
       inner join (SELECT t.ct AS CONTRACTOR_ID,
                          t.cf AS TYPE_ID
                   FROM   sf_claslink t,
                          executor e,
                          merop m
                   WHERE  t.ct = e.id AND t.cf in {types}
                          AND t.cf = m.id) AS meropContract
               ON meropContract.contractor_id = executorsbyorg.id)             
               
select executors.*,objects.well_id,objects.well_pad_id, objects.field_id,objects.shop_id,enterprise_id from executors  inner join objects on executors.id=objects.executor_id 

WHERE (objects.shop_id is null) or objects.shop_id in {shops}