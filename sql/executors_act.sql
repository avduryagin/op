WITH contractdata
     AS (SELECT c.id       contract,
                m.class_id meropType,
                c.executor,
                c.start_date,
                c.end_date
         FROM   sf_contract c,
                sf_contract_cost m
         WHERE  C.id = m.contract_id
                AND c.end_date > '2021-01-31'),
     contractstoshop
     AS (SELECT *
         FROM   sf_contract_shop scs
                inner join contractdata
                        ON scs.contract_id = contractdata.contract
         WHERE  shop_id = 323938109),
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
                       ON c.parent_id = c2.id),
     merop
     AS (SELECT c.*
         FROM   sf_class c,
                sf_metalink m
         WHERE  c.sf_ref_type = m.ct
                AND m.cf = 'MEROP'),

   q as (
select * from
(select *  from sf_activity a
 where exists (select null from sf_activity_plan p where p.activity_id = a.id and p.plan_id = 343043 and p.start_date <= cast('01.01.2023' as date) and p.end_date >= cast('01.01.2022' as date) and p.distributed = 1) 
   and a.visible = 1
          AND a.id IN( (SELECT o.activity_id from sf_activity_own o WHERE o.entity_id = 323938109 AND o.entity_name = 'SHOP') intersect (SELECT o.activity_id from sf_activity_own o WHERE o.entity_id = 1200186503 AND o.entity_name = 'FIELD') intersect (SELECT o.activity_id from sf_activity_own o WHERE o.entity_id = 1200186103 AND o.entity_name = 'ENTERPRISE') )
  
union all
select *
  from sf_activity a
 where exists (select null from sf_activity_plan p where p.activity_id = a.id and p.plan_id = 343043 and p.start_date <= cast('01.01.2023' as date) and p.end_date >= cast('01.01.2022' as date) and 1 = 0 and p.distributed = 0)
    and a.visible = 1
          AND a.id IN( (SELECT o.activity_id from sf_activity_own o WHERE o.entity_id = 323938109 AND o.entity_name = 'SHOP') intersect (SELECT o.activity_id from sf_activity_own o WHERE o.entity_id = 1200186503 AND o.entity_name = 'FIELD') intersect (SELECT o.activity_id from sf_activity_own o WHERE o.entity_id = 1200186103 AND o.entity_name = 'ENTERPRISE') )
    
 ) iv
UNION ALL
select *
  from sf_activity a
 where coalesce(a.start_date, to_date('01.01.2100', 'dd.mm.yyyy')) <= cast('01.01.2023' as date) and coalesce(a.end_date, to_date('01.01.2100', 'dd.mm.yyyy')) >= cast('01.01.2022' as date)
          AND a.id IN( (SELECT o.activity_id from sf_activity_own o WHERE o.entity_id = 323938109 AND o.entity_name = 'SHOP') intersect (SELECT o.activity_id from sf_activity_own o WHERE o.entity_id =1200186503 AND o.entity_name = 'FIELD') intersect (SELECT o.activity_id from sf_activity_own o WHERE o.entity_id = 1200186103 AND o.entity_name = 'ENTERPRISE') )
   and a.visible = 1),
   
   wells as (SELECT exp.WELL_ID,
       exp.WELL_CODE,
       exp.COOP_ID,
       exp.ENTERPRISE_ID,
       exp.CITS_ID,
       exp.SHOP_ID,
       exp.CREW_ID,
       exp.FIELD_ID,
       exp.FIELD_CODE,
       coalesce(field_1485.identifier, exp.FIELD_NAME) FIELD_NAME,
       coalesce(field_1487.identifier, exp.FIELD_NAME) FIELD_NAME_EN,
       exp.POOL_ID,
       exp.WELL_PAD_ID,
       coalesce(wellpad_1485.identifier, EXP.WELL_PAD_NAME) WELL_PAD_NAME,
       coalesce(wellpad_1487.identifier, EXP.WELL_PAD_NAME) WELL_PAD_NAME_EN,
       coalesce(well_1485.identifier, exp.WELL_NAME) WELL_NAME,
       coalesce(well_1487.identifier, exp.WELL_NAME) WELL_NAME_EN,
       exp.STATUS_CODE,
       exp.PURPOSE_ID,
       exp.PURPOSE_CODE,
       exp.OPERATION_METHOD_CODE,
       exp.PODST_ID,
       exp.FEEDER_ID,
       exp.KTP_ID,
       exp.DNS_ID,
       exp.GZU_ID,
       exp.OTVOD_GZU_ID,
       exp.KNS_ID,
       exp.VRP_ID,
       exp.OTVOD_VRP_ID,
       exp.WP_REGIME_CURRENT,
       exp.OP_FOND,
       exp.DENSITY_OIL_REGIME_CURRENT,
       exp.BEHAVIOR_CODE,
       coord.y as LATITUDE,
       coord.x as LONGITUDE
  FROM "EXP#WELL_FOND" as exp
    LEFT JOIN "V#EARTH_SURF_FEATURE_ALIAS" as field_1485 ON field_1485.earth_surf_feature_s=exp.FIELD_ID AND field_1485.naming_sys_id=1485
    LEFT JOIN "V#EARTH_SURF_FEATURE_ALIAS" as field_1487 ON field_1487.earth_surf_feature_s=exp.FIELD_ID AND field_1487.naming_sys_id=1487
    LEFT JOIN "V#FACILITY_ALIAS" as wellpad_1485 ON wellpad_1485.aliased_object_s=exp.WELL_PAD_ID AND wellpad_1485.aliased_object_t='WELL_PAD' AND wellpad_1485.naming_sys_id=1485
    LEFT JOIN "V#FACILITY_ALIAS" as wellpad_1487 ON wellpad_1487.aliased_object_s=exp.WELL_PAD_ID AND wellpad_1487.aliased_object_t='WELL_PAD' AND wellpad_1487.naming_sys_id=1487
    LEFT JOIN "V#FACILITY_ALIAS" as well_1485 ON well_1485.aliased_object_s=exp.WELL_ID AND well_1485.aliased_object_t='WELL' AND well_1485.naming_sys_id=1485
    LEFT JOIN "V#FACILITY_ALIAS" as well_1487 ON well_1487.aliased_object_s=exp.WELL_ID AND well_1487.aliased_object_t='WELL' AND well_1487.naming_sys_id=1487
    LEFT JOIN p_well_coord coord on coord.well_id = exp.well_id and start_time <= now() and coalesce (end_time, cast('01.01.2100' as date)) > now() and coord.r_coord_sys_cnstr_id = 7664772222)  


select executors.id as execId,executors.name,merops.id as meropId, merops.type_id, merops.obj_type_code, merops.obj_id, wells.WELL_ID, wells.WELL_PAD_ID, wells.FIELD_ID, wells.SHOP_ID, wells.WELL_PAD_NAME 
from 
(SELECT meropContract.merop_id,
       executorsbyorg.* from executorsbyorg
       inner join (SELECT t.ct AS CONTRACTOR_ID,
                          t.cf AS MEROP_ID
                   FROM   sf_claslink t,
                          executor e,
                          merop m
                   WHERE  t.ct = e.id
                          AND t.cf = m.id) AS meropContract
               ON meropContract.contractor_id = executorsbyorg.id) executors
               right join (select * from VSF_ACTIVITY_OPT_PLAN v
 WHERE     v.PLAN_ID = 343043
       AND id in (SELECT /*+ cardinality (t,1) */q.id  FROM  q)) merops 
       inner join wells on wells.WELL_ID = merops.obj_id and merops.obj_type_code = 'WL'
       on merops.type_id = executors.merop_id
       where executors.id is null or executors.id in (select e.executor_id from sf_executor_object e 
       where (e.well_pad_id is not null and e.well_pad_id = wells.WELL_PAD_ID)
       or (e.well_pad_id is null and e.field_id is not null and e.field_id = wells.FIELD_ID )
       or (e.well_pad_id is null and e.field_id is null and e.shop_id is not null and e.shop_id = wells.shop_id) )
       order by execid desc