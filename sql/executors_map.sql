       WITH executor
     AS (SELECT c.ID,
                c.SF_REF_TYPE,
                COALESCE (c2.NAME, c.NAME) as   ORGNAME,
                COALESCE (c2.NAME, c.NAME, ' ') as NAME,
                c.CODE,
                c.PARENT_ID
           FROM sf_class c
                INNER JOIN sf_metalink m ON m.cf = 'EXECUTOR' AND m.ct = c.sf_ref_type
                LEFT JOIN sf_class c2 ON c.parent_id = c2.id),
     merop
     AS (SELECT c.*
           FROM sf_class c, sf_metalink m
          WHERE c.sf_ref_type = m.ct AND m.cf = 'MEROP')
SELECT t.ct AS CONTRACTOR_ID,
       t.cf AS MEROP_ID,
       e.PARENT_ID
  FROM sf_claslink t, executor e, merop m
 WHERE t.ct = e.id AND t.cf = m.ID