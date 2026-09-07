# Claim Submission & Escalation Guide

**MediAssist Health Network — Billing & Insurance Operations**
Document ref: BILL-OPS-011 · Version 6.0 · Owner: Central Billing Office, Hyderabad
Audience: Billing Executives, Floor Coordinators, Admin · Access: `billing_executive`, `admin`

---

## Purpose & Scope

This guide is the standard operating reference for billing executives handling insurance
claims at any MediAssist hospital or clinic. It covers the four claim journeys you will
encounter daily — **cashless pre-authorisation**, **reimbursement**, **pre-authorisation
enhancement**, and **rejection response** — plus the escalation matrix and fraud-prevention
controls that govern all of them.

It applies to every empanelled insurer listed in `billing_codes.pdf`. Where an
insurer-specific portal or Third Party Administrator (TPA) differs from the general process,
the difference is called out explicitly.

> **Golden rule:** Every claim action must leave an audit trail. If it is not logged in the
> MediAssist Billing Portal (MBP), for claim purposes it did not happen.

### How claims flow at MediAssist

```
Admission ──► Eligibility check ──► Pre-auth (cashless) ──► Treatment ──► Final bill
                  │                        │                                  │
                  └─ not eligible ─► Reimbursement route        Enhancement (if bill > approved)
                                                                              │
                                          Approved ◄── Query/Rejection ◄──────┘
```

---

## 1. Cashless Claim Process

Cashless is the default pathway for **planned and emergency admissions** where the patient
holds a policy with an empanelled insurer. The hospital is paid directly by the insurer/TPA;
the patient pays only non-covered items, co-pay and amounts above sub-limits.

### 1.1 Pre-authorisation timeline

| Admission type | Pre-auth request deadline |
|---|---|
| Planned admission | At least **48 hours before** admission |
| Emergency admission | **Within 6 hours** of admission |

Missing the 6-hour emergency window is the single most common reason for avoidable cashless
denials. Raise the request even with provisional details and enhance later (see Section 3).

### 1.2 Documents required for pre-authorisation

1. Duly filled insurer-specific pre-authorisation form (download from the insurer portal).
2. Admission note with **provisional diagnosis** and matching ICD-10 code from `billing_codes.pdf`.
3. Treating doctor's **treatment plan** and estimated length of stay (LOS).
4. Cost estimate mapped to the relevant **package rate**.
5. Patient KYC: policy card / e-card, government photo ID, and policy number.
6. For accident cases: MLC (Medico-Legal Case) number and FIR copy where applicable.

### 1.3 Step-by-step

1. Verify patient eligibility on the insurer portal (policy active, sum insured available,
   waiting period cleared, room-rent category).
2. Counsel the patient on room eligibility — exceeding the eligible room rent triggers a
   **proportionate deduction** across the whole bill.
3. Create the claim in **MBP → Claims → New Cashless** and attach the documents above.
4. Submit the pre-auth request via the insurer portal **and** log the portal reference number
   back in MBP.
5. Record the timestamp of submission — the SLA clock starts here.
6. Track status hourly until an approval, query, or denial is received.
7. On approval, record the **approved amount** and **pre-auth number** in MBP and inform the ward.
8. On a query, respond within **2 hours** with the requested clinical documents.

### 1.4 Typical approval turnaround (initial pre-auth)

| Insurer | Standard SLA | Notes |
|---|---|---|
| Star Health | 2–4 hrs | Fastest; auto-adjudication for standard packages |
| HDFC Ergo | 3–6 hrs | Frequent clinical queries on cardiac packages |
| ICICI Lombard | 3–6 hrs | TPA-routed; confirm TPA before submitting |
| New India Assurance | 4–8 hrs | PSU insurer; slower on weekends |
| United India | 4–8 hrs | Manual review common |
| Bajaj Allianz | 3–5 hrs | |
| Niva Bupa | 2–5 hrs | |
| Care Health | 3–6 hrs | |

**If no response within the SLA window**, escalate per the matrix in Section 5.

### 1.5 Worked example

> A patient is admitted at 9:00 pm with chest pain; provisional diagnosis **NSTEMI (I21.4)**,
> package rate **₹1,20,000**, insurer **HDFC Ergo**. The executive raises the pre-auth by
> 11:30 pm (within the 6-hour window), attaching the admission note, ECG and troponin.
> HDFC Ergo raises a clinical query on the troponin trend at 1:00 am; it is answered within
> the 2-hour window. Approval for ₹1,10,000 (with ₹10,000 towards non-payables) lands at 4:00 am.
> Total turnaround: ~4.5 hours — within SLA.

---

## 2. Reimbursement Claim Process

Used when the patient **pays the hospital directly** and claims from the insurer afterwards —
typically when the insurer is not empanelled, the patient chose not to use cashless, or a
cashless request was declined for administrative (not clinical) reasons.

### 2.1 Documents required

1. Original itemised hospital bills with payment receipts.
2. **Discharge summary** signed by the treating consultant.
3. Prescription copies and pharmacy invoices.
4. Investigation reports (lab, radiology) supporting the diagnosis.
5. Completed reimbursement claim form with patient **NEFT/bank details**.
6. Cancelled cheque or passbook copy for payout.
7. KYC documents and policy copy.

### 2.2 Step-by-step

1. Counsel the patient at discharge that the claim is reimbursement, not cashless.
2. Issue the complete original document set and retain certified copies in MBP.
3. Help the patient fill the claim form; verify the ICD-10 and procedure codes match the bills.
4. Note the insurer's **submission deadline** — typically **30 days post-discharge** (confirm
   per insurer; some allow 15, some 90).
5. Hand over a document checklist so the patient can track what they submitted, and send a
   written (email/SMS) reminder of the deadline.

> Reimbursement claims are owned by the patient once they leave, but billing executives are
> responsible for issuing a **complete, claim-ready document set**. Incomplete discharge
> paperwork is a recurring rejection cause.

---

## 3. Pre-Authorisation Enhancement

Raise an **enhancement** when the initially approved amount will not cover the actual
treatment cost — e.g. the patient needed ICU escalation, an additional procedure, or a
longer LOS than estimated.

### 3.1 When to raise

- Projected final bill exceeds the approved pre-auth amount by more than **10%**.
- A new procedure or implant (e.g. stent, prosthesis) is added mid-admission.
- LOS extends beyond the approved package days.

### 3.2 Documents required

1. Enhancement request form referencing the **original pre-auth number**.
2. **Clinical justification** note from the treating doctor explaining the change.
3. Revised cost estimate and updated treatment plan.
4. Any new investigation reports supporting the escalation.

### 3.3 Process

1. Raise the enhancement **before** the original approved amount is exhausted — never after
   discharge.
2. Submit via the insurer portal linked to the original pre-auth, clearly marked "Enhancement".
3. Track as a child claim under the parent pre-auth in MBP.
4. If the enhancement is delayed beyond **6 hours** and the patient is due for discharge,
   escalate to the insurer nodal officer (Section 5).

---

## 4. Claim Rejection Response

A rejection is not the end of the claim. Most rejections on **administrative or documentation
grounds** are reversible on appeal. Rejection codes here map to the exclusion codes in
`billing_codes.pdf`.

### 4.1 Common rejection codes

| Code | Reason | First response |
|---|---|---|
| EXCL-01 | Non-disclosure of pre-existing condition | Submit prior medical records / declaration |
| EXCL-02 | Treatment within waiting period | Verify policy dates; appeal if mis-calculated |
| EXCL-03 | Non-empanelled treating doctor / provider | Provide registration proof; or move to reimbursement |
| EXCL-04 | Day-care procedure claimed as inpatient | Re-submit under correct day-care package |
| EXCL-05 | Cosmetic / excluded procedure | Usually final; confirm exclusion in policy |
| EXCL-07 | Investigation-only admission | Provide clinical necessity note |
| EXCL-08 | Documentation beyond deadline | Appeal with justification for delay |

### 4.2 Counter-response template

> **Subject:** Reconsideration request — Claim [CLM-2024-XXXX] / Pre-auth [number]
>
> Dear [Insurer/TPA],
> We request reconsideration of the above claim rejected under code [EXCL-0X] dated [date].
> Please find attached [document] which addresses the stated reason. The treating consultant's
> clarification is enclosed. We request review within the reconsideration window.
> Regards, MediAssist Billing — [Campus].

### 4.3 Deadlines

- Reconsideration / appeal must be filed within the insurer's window — **usually 90 days** from
  rejection.
- If the appeal is denied, escalate to the **insurer grievance cell**, then the **Insurance
  Ombudsman** (Section 5).

---

## 5. Escalation Matrix

Use this matrix when an SLA is breached or a claim stalls. Always escalate **in writing** and
log each step in MBP.

| Scenario | First Contact | Escalation (24 hrs no response) | Final Escalation |
|---|---|---|---|
| Cashless pre-auth delayed | TPA helpline | Insurer nodal officer | MediAssist billing manager |
| Claim rejected — clinical grounds | Medical reviewer call | Written appeal with CMO sign-off | Insurance ombudsman |
| Payment delayed >30 days post-discharge | Accounts team | CFO office | Legal cell |
| Enhancement not approved before discharge | TPA helpline | Insurer nodal officer | MediAssist billing manager |

**Escalation discipline:**

- Wait the stated window before moving to the next level — premature escalation resets goodwill.
- Every escalation email must quote the claim ID, pre-auth number, patient ID, and the SLA breached.
- CC the campus billing manager on any escalation beyond first contact.
- Never let more than **24 hours** pass without an escalation when a response is overdue.

---

## 6. Fraud Prevention

Claim fraud — internal or external — is a zero-tolerance area governed by the Code of Conduct.
Billing executives are the first line of defence.

### 6.1 Warning signs

- Mismatch between the diagnosis, the procedure, and the investigation reports.
- Inflated or duplicated line items on the bill.
- Documents that appear altered, back-dated, or inconsistent in handwriting/fonts.
- Pressure from any party to "adjust" codes to a higher package rate (up-coding).
- Repeated claims for the same patient across facilities within a short period.

### 6.2 Controls

- **Mandatory dual-check** for any claim above **₹2,00,000** — a second billing executive must
  independently verify codes, amounts, and documents before submission.
- Every claim must carry a complete **audit trail** in MBP: who created, edited, submitted, and
  approved each step, with timestamps.
- Claim files are retained for a minimum of **7 years**.
- Never share insurer portal credentials; each executive uses their own login.

### 6.3 Reporting

Suspected fraud must be reported to the **Billing Manager and Internal Audit within 24 hours**
via MBP → Compliance → Report Concern (or compliance@mediassist.in). Reports may be made
confidentially. Retaliation against good-faith reporters is itself a disciplinary offence.

---

## 7. Key Performance Indicators (KPIs)

Billing executives are measured monthly on the following. Targets are indicative.

| KPI | Target | Why it matters |
|---|---|---|
| Pre-auth raised within SLA | > 95% | Drives same-day approvals |
| Query response time | < 2 hrs | Prevents avoidable rejections |
| First-pass approval rate | > 85% | Reflects coding & documentation quality |
| Claim rejection rate | < 10% | Quality and compliance indicator |
| Average days to payment | < 21 days | Cash-flow health |
| Reopened/appealed claims won | > 60% | Effectiveness of escalation |

---

## Quick Reference Card

| Action | Deadline / SLA |
|---|---|
| Emergency cashless pre-auth | Within 6 hrs of admission |
| Planned cashless pre-auth | 48 hrs before admission |
| Response to insurer query | Within 2 hrs |
| Reimbursement submission | Within 30 days of discharge |
| Rejection reconsideration | Within 90 days of rejection |
| Dual-check threshold | Claims > ₹2,00,000 |
| Fraud report | Within 24 hrs of suspicion |
| Escalation cadence | No gap > 24 hrs when overdue |

---

*For code lookups (ICD-10, procedures, package rates), the full insurer panel with portal URLs
and TPA contacts, room-rent sub-limits and exclusion codes, see `billing_codes.pdf`. For
escalation related to staff conduct, see `code_of_conduct.pdf`.*
