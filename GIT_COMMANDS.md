# คำสั่ง Git/GitHub พื้นฐานสำหรับโปรเจกต์นี้

รวมคำสั่งที่ใช้ซ้ำบ่อยที่สุดในการทำงานกับโปรเจกต์นี้ เปิดไฟล์นี้ดูได้เลยแทนที่จะถามซ้ำทุกครั้ง

ทุกคำสั่งรันจาก root ของโปรเจกต์:
```bash
cd "/Users/athens/Downloads/Coding/Telco Customer Churn"
```

## เช็คสถานะก่อนทำอะไรก็ตาม

```bash
git status
```
ดูว่าไฟล์ไหนแก้ไปแล้วบ้าง ไฟล์ไหนยังไม่ได้ commit — ควรรันคำสั่งนี้ก่อนเสมอเพื่อดูภาพรวมก่อนตัดสินใจ

## ขั้นตอนหลัก: ส่งงานที่แก้ไขขึ้น GitHub (ใช้บ่อยที่สุด)

```bash
git add .                          # เก็บไฟล์ที่แก้ไขทั้งหมดเข้าคิว (staging)
git commit -m "อธิบายว่าแก้อะไร"    # บันทึกเป็นเวอร์ชันใหม่พร้อมข้อความอธิบาย
git push                           # ส่งขึ้น GitHub
```

ถ้าอยากเลือก add เฉพาะบางไฟล์แทน `git add .` ทั้งหมด:
```bash
git add README.md telco_churn_eda.ipynb
```

## รูปแบบการเขียน Commit Message (มาตรฐานสากล: Conventional Commits)

```
<type>(<scope>): <คำอธิบายสั้นๆ>

<รายละเอียดเพิ่มเติม (ถ้ามี)>
```

**ประเภท (type) ที่ใช้บ่อย:**

| Type | ใช้เมื่อ | ตัวอย่าง |
|---|---|---|
| `feat` | เพิ่มฟีเจอร์ใหม่ | `feat: add step 10 customer risk segmentation` |
| `fix` | แก้บั๊ก | `fix: correct churn rate calculation in step 5` |
| `docs` | แก้เอกสาร (README ฯลฯ) | `docs: update readme methodology for step 8-9` |
| `refactor` | ปรับโค้ดโดยไม่เปลี่ยนผลลัพธ์ | `refactor: simplify tenure bucket logic` |
| `chore` | งานจุกจิก ไม่กระทบโค้ดหลัก | `chore: add scikit-learn to requirements.txt` |
| `style` | จัดฟอร์แมต ไม่กระทบ logic | `style: fix thai font rendering in charts` |
| `test` | เพิ่ม/แก้ test | `test: verify notebook runs end to end` |

**กติกาสั้นๆ:**
1. บรรทัดแรก (subject) ไม่เกิน ~50-72 ตัวอักษร ใช้ **imperative mood** (สั่งให้ทำ เช่น `add` ไม่ใช่ `added`/`adds`)
2. ตัวพิมพ์เล็กทั้งหมดใน type/scope และไม่ต้องมีจุดท้ายประโยค
3. ถ้าต้องอธิบายเพิ่ม เว้นบรรทัดว่าง 1 บรรทัดแล้วเขียนต่อ — เน้นอธิบาย "ทำไม" ถึงแก้ ไม่ใช่ "อะไร" (โค้ด diff บอกอยู่แล้วว่าแก้อะไร)

**ตัวอย่างการใช้กับ `git commit`:**
```bash
git commit -m "feat: add step 5 categorical vs churn analysis"
```

## ดูประวัติการเปลี่ยนแปลง

```bash
git log --oneline          # ดูประวัติ commit แบบย่อ (บรรทัดเดียวต่อ 1 commit)
git log --oneline -10      # ดูแค่ 10 commit ล่าสุด
git diff                   # ดูว่าไฟล์ที่ยังไม่ได้ commit เปลี่ยนตรงไหนบ้าง (บรรทัดที่เพิ่ม/ลบ)
```

## ดึงงานล่าสุดจาก GitHub มาที่เครื่อง

```bash
git pull
```
ใช้เวลาทำงานจากหลายเครื่อง หรือมีคนอื่นแก้ไขโปรเจกต์นี้ด้วย เพื่อดึงเวอร์ชันล่าสุดมาก่อนเริ่มแก้ต่อ

## แก้ไขผิดพลาด (ใช้เท่าที่จำเป็น)

```bash
git restore <ชื่อไฟล์>       # ยกเลิกการแก้ไขที่ยังไม่ได้ commit ของไฟล์นั้น (คืนกลับเป็นเวอร์ชันล่าสุดที่ commit ไว้)
git reset HEAD~1 --soft     # ยกเลิก commit ล่าสุด แต่ยังเก็บการแก้ไขไว้ (ไม่ลบงาน)
```
**ข้อควรระวัง:** `git restore` จะลบการแก้ไขที่ยังไม่ได้ commit ทิ้งถาวร ใช้ให้แน่ใจก่อนว่าไม่เสียดายงานที่แก้ไว้

## กรณีต้องตั้งค่าใหม่ตั้งแต่ต้น (ทำไปแล้วครั้งแรก ไว้เผื่ออ้างอิง)

```bash
git init                                                  # เริ่มต้น git repo ในโฟลเดอร์นี้
gh repo create <ชื่อ-repo> --public --source=. --remote=origin --push   # สร้าง repo บน GitHub + push ในคำสั่งเดียว (ต้องมี GitHub CLI)
```

หรือถ้าสร้าง repo ผ่านหน้าเว็บ GitHub ไว้แล้ว:
```bash
git remote add origin https://github.com/<username>/<repo-name>.git
git branch -M main
git push -u origin main
```

## หมายเหตุ

- ถ้า push แล้วเจอ error เรื่อง authentication ให้ใช้ Personal Access Token แทน password (GitHub เลิกรองรับ password login แล้ว) หรือใช้ SSH key แทน HTTPS
- ไฟล์ `.claude/settings.local.json` และ `.venv/` ถูกตั้งไว้ใน `.gitignore` แล้ว จะไม่ถูก push ขึ้นไปโดยอัตโนมัติ (เป็นความตั้งใจ ไม่ใช่ error)
