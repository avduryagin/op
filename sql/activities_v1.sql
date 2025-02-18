with data as (
with wells as(
with ci as(
with frame as(
with ID as (
          
{objects}),      
          
  
           q as (
select * from
(select *  from sf_activity a
 where exists (select null from sf_activity_plan p where p.activity_id = a.id and p.plan_id = {plan_id} and p.start_date <= cast(to_date('{edate}','dd.mm.yyyy') as date) and p.end_date >= cast(to_date('{sdate}','dd.mm.yyyy') as date) and (p.distributed = 1 or (1 = 0 and p.distributed = 0))) 
   and a.visible = 1
          AND a.id IN(select * from ID)
 ) iv
UNION ALL
select *
  from sf_activity a
 where coalesce(a.start_date, to_date('01.01.2100', 'dd.mm.yyyy')) <= cast(to_date('{edate}','dd.mm.yyyy') as date) and coalesce(a.end_date, to_date('01.01.2100', 'dd.mm.yyyy')) >= cast(to_date('{sdate}','dd.mm.yyyy') as date)
          AND a.id IN(select * from ID)
   and a.visible = 1) 

select x.*,sc.id as "class_id",sc.code as "class_code",sc.parent_id  as "class_parent_id",sc.sf_ref_type,sc.name as "class_name"
  FROM (select * from VSF_ACTIVITY_OPT_PLAN v
 WHERE     v.PLAN_ID = {plan_id}
       AND id in (SELECT /*+ cardinality (t,1) */q.id  FROM  q)
) x 
inner join sf_class sc on x.type_id = sc.id)

select
frame.id,
frame.obj_id as "well",
frame.obj_type_code as "object_type",
frame.oil_production_after as "Q1",
frame.oil_production_before as "Q0",
frame.start_date_min as "supp0",
frame.start_date_max as "supp1",
frame.start_date as "begin",
frame.end_date as "end",
frame.plan_start_date as "plan_begin",
frame.plan_end_date as "plan_end",
frame.work_time as "duration",
frame.plan_start_date_wellstop as "well_down",
frame.plan_end_date_wellstop as "well_up",
frame.wellstop_time as "wellstop_duration",
frame.class_id as "act_type",
frame.class_code as "act_code",
frame.class_parent_id as "parent_id"
from frame)

select ci.*,
coord.x as LATITUDE,
coord.y as LONGITUDE
from ci
left join p_well_coord coord
on ci.well=coord.well_id
and coord.r_coord_sys_cnstr_id = 7664772222 and start_time <= now() and coalesce (end_time, cast('01.01.2100' as date)) > now() and coord.r_coord_sys_cnstr_id = 7664772222 
)


select wells.*,g.well_pad_id,g.field_id,g.shop_id,g.ENTERPRISE_ID 
from wells
left join "EXP#WELL_FOND" g
on wells.well=g.WELL_ID
)
select data.*,tuple.cortege_activity_id,tuple.position, tuple.begin_from,tuple.begin_to,tuple.from_end
from data
left join SF_ACTIVITY_CORTEGE_ITEM tuple
on data.id=tuple.activity_id    