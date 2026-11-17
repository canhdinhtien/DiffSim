import json,sys
r=json.load(open("out/01_profile.json",encoding="utf-8"))
for t in sys.argv[1:]:
    v=r[t]; print("=====",t,v["rows_profiled"])
    for c,d in v["columns"].items():
        dd={k:x for k,x in d.items() if k not in("type","std","q25","q75")}
        if "top" in dd: dd["top"]=dd["top"][:6]
        print(f"  {c} [{d['type']}] {json.dumps(dd,ensure_ascii=False)[:300]}")
