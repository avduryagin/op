with p as(
with q as(
SELECT 
       R.RULE_ID,
       l.CF,
       l.CT,
       r.val hours,
       ro.enterprise_id,
       ro.cits_id,
       ro.shop_id,
       ro.field_id,
       ro.well_pad_id,
       ro.well_id 
  FROM SF_CLASLINK l
       INNER JOIN SF_RULE r ON l.id = r.claslink_id
       LEFT JOIN sf_rule_object ro on ro.rule_id = r.id 
       INNER JOIN sf_class c1 ON l.cf = c1.id
       INNER JOIN sf_class c2 ON l.ct = c2.id
       INNER JOIN sf_class c3 ON c3.id = R.RULE_ID AND c3.sf_ref_type = 'RMRULE' AND c3.parent_id IS NOT NULL 
       where r.rule_id ={rule})
select q.rule_id,q.cf,q.ct,q.hours from q where q.cf in {type_id_tuple})
select * from p where p.ct in {type_id_tuple}