const API = "http://127.0.0.1:8000";

document.querySelectorAll(".nav-btn").forEach(btn=>{
  btn.onclick = ()=>{
    document.querySelectorAll(".nav-btn").forEach(b=>b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p=>p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.panel).classList.add("active");
  };
});

// Pear Count
document.getElementById("countForm").addEventListener("submit", async (e)=>{
  e.preventDefault();
  const f = document.getElementById("countImg").files[0];
  if(!f) return;
  const fd = new FormData();
  fd.append("file", f);
  fd.append("conf", "0.25");
  const res = await fetch(API+"/api/count", {method:"POST", body:fd});
  const data = await res.json();
  if(data.error){ alert(data.error); return; }
  document.getElementById("countNum").textContent = data.count;
  const bytes = new Uint8Array(data.image_b64.match(/.{1,2}/g).map(h=>parseInt(h,16)));
  const url = URL.createObjectURL(new Blob([bytes], {type:"image/jpeg"}));
  document.getElementById("countPreview").src = url;
});

// Disease
document.getElementById("diseaseForm").addEventListener("submit", async (e)=>{
  e.preventDefault();
  const f = document.getElementById("leafImg").files[0];
  if(!f) return;
  const fd = new FormData(); fd.append("file", f);
  const res = await fetch(API+"/api/disease", {method:"POST", body:fd});
  const data = await res.json();
  if(data.error){ alert(data.error); return; }
  document.getElementById("diseaseClass").textContent = data.top_class;
  document.getElementById("probs").textContent = JSON.stringify(data.probs, null, 2);
});

// Boxes (Yield)
document.getElementById("estimateBtn").onclick = async ()=>{
  const detected = +document.getElementById("detCount").value;
  const ppb = +document.getElementById("ppb").value;
  const recall = +document.getElementById("recall").value;
  const res = await fetch(API+"/api/boxes", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({detected_count:detected, pears_per_box:ppb, detection_recall:recall})
  });
  const data = await res.json();
  if(data.error){ alert(data.error); return; }
  document.getElementById("boxesOut").textContent = data.estimated_boxes;
};

// Spray
document.getElementById("fetchSpray").onclick = async ()=>{
  const stage = document.getElementById("stage").value;
  const res = await fetch(API+"/api/spray?stage="+stage);
  const data = await res.json();
  const el = document.getElementById("sprayList");
  el.innerHTML = "";
  (data.recommendations || []).forEach(block=>{
    const h = document.createElement("h3"); h.textContent = block.stage.replace("_"," ");
    el.appendChild(h);
    const ul = document.createElement("ul");
    block.guidelines.forEach(g=>{ const li=document.createElement("li"); li.textContent=g; ul.appendChild(li); });
    el.appendChild(ul);
  });
};
