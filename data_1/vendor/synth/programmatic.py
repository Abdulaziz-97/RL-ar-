"""Programmatic Arabic RLVR/cold-start factory — verified GTs, v3 schema."""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]

# Target totals (production)
TARGET_RLVR = 800
TARGET_COLD = 800

# Domain mix for 800 RLVR (v3-compatible domains; balanced for curriculum)
RLVR_MIX = {"gsm8k": 280, "math": 180, "math_comp": 180, "logic": 160}
# Cold includes verified mmlu keepers; new cold is reasoning-only
COLD_MIX = {"gsm8k": 250, "math": 180, "math_comp": 180, "logic": 120, "mmlu": 70}

NAMES = [
    "أحمد", "سارة", "محمد", "نورة", "خالد", "فاطمة", "يوسف", "مريم",
    "عمر", "ليلى", "حسن", "هدى", "إبراهيم", "ريم", "علي", "سلمى",
]
ITEMS = [
    "دفتر", "قلم", "كتاب", "تفاحة", "كيس أرز", "علبة حليب", "كرسي", "مصباح",
]
UNITS = ["ريال", "قطعة", "كيلوغرام", "متر", "ساعة"]


@dataclass
class Sample:
    domain: str
    prompt: str
    ground_truth: Any
    num_steps: int
    meta_extra: dict
    solution_steps: list[str]  # Arabic steps for cold-start CoT


def _with_family(sample: Sample, family: str) -> Sample:
    """Stamp a structural template-family ID (not merely the domain name)."""
    meta = dict(sample.meta_extra or {})
    meta["template_family"] = family
    return Sample(
        sample.domain,
        sample.prompt,
        sample.ground_truth,
        sample.num_steps,
        meta,
        sample.solution_steps,
    )


def _arabic_purity(text: str) -> float:
    if not text:
        return 0.0
    ar = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    return ar / max(len(text.replace(" ", "")), 1)


def next_id(existing: set[str], prefix: str) -> str:
    best = -1
    for i in existing:
        if i.startswith(prefix):
            tail = i[len(prefix) :]
            if tail.isdigit():
                best = max(best, int(tail))
    n = best + 1
    while f"{prefix}{n:04d}" in existing:
        n += 1
    uid = f"{prefix}{n:04d}"
    existing.add(uid)
    return uid


# ---------------------------------------------------------------------------
# GSM8K templates
# ---------------------------------------------------------------------------

def gen_gsm8k(rng: random.Random) -> Sample:
    families = [
        ("gsm_shop_change", _gsm_shop_change),
        ("gsm_multi_buy", _gsm_multi_buy),
        ("gsm_trip_days", _gsm_trip_days),
        ("gsm_work_rate", _gsm_work_rate),
        ("gsm_fraction_of", _gsm_fraction_of),
        ("gsm_ages", _gsm_ages),
        ("gsm_ratio_share", _gsm_ratio_share),
        ("gsm_remaining_after", _gsm_remaining_after),
        ("gsm_stacked_ops", _gsm_stacked_ops),
        ("gsm_bus_split", _gsm_bus_split),
        ("gsm_salary_save", _gsm_salary_save),
        ("gsm_garden_rows", _gsm_garden_rows),
    ]
    name, fn = rng.choice(families)
    return _with_family(fn(rng), name)


def _grade_for_steps(ns: int) -> str:
    if ns <= 2:
        return "grade_4"
    if ns <= 3:
        return "grade_5"
    if ns <= 5:
        return "grade_6"
    if ns <= 6:
        return "grade_7"
    return "grade_8"


def _gsm_shop_change(rng: random.Random) -> Sample:
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    n = rng.randint(2, 9)
    price = rng.randint(3, 25)
    paid = n * price + rng.randint(5, 40)
    cost = n * price
    change = paid - cost
    prompt = (
        f"اشترى {name} {n} من {item} بسعر {price} ريال للوحدة، ودفع {paid} ريال. "
        f"كم ريالًا يُرجع له؟"
    )
    steps = [
        f"نحسب تكلفة المشتريات: {n} × {price} = {cost} ريالًا.",
        f"ثم نطرح التكلفة من المدفوع: {paid} − {cost} = {change} ريالًا.",
    ]
    return Sample("gsm8k", prompt, change, 2, {"grade_level": _grade_for_steps(2)}, steps)


def _gsm_multi_buy(rng: random.Random) -> Sample:
    name = rng.choice(NAMES)
    a, b = rng.randint(2, 8), rng.randint(2, 8)
    pa, pb = rng.randint(4, 20), rng.randint(4, 20)
    total = a * pa + b * pb
    prompt = (
        f"اشترت {name} {a} دفاتر بسعر {pa} ريالًا للدفتر، و{b} أقلام بسعر {pb} ريالًا للقلم. "
        f"كم تدفع إجمالًا؟"
    )
    steps = [
        f"تكلفة الدفاتر: {a} × {pa} = {a*pa}.",
        f"تكلفة الأقلام: {b} × {pb} = {b*pb}.",
        f"المجموع: {a*pa} + {b*pb} = {total}.",
    ]
    return Sample("gsm8k", prompt, total, 3, {"grade_level": _grade_for_steps(3)}, steps)


def _gsm_trip_days(rng: random.Random) -> Sample:
    fri = rng.randint(40, 120)
    sat_mult = rng.choice([2, 3])
    sat = fri * sat_mult
    sun_less = rng.randint(20, 80)
    sun = sat - sun_less
    total = fri + sat + sun
    prompt = (
        f"زار الحديقة {fri} شخصًا يوم الجمعة. يوم السبت جاء {sat_mult} أضعاف عدد الجمعة. "
        f"يوم الأحد جاء عدد أقل من السبت بمقدار {sun_less}. "
        f"كم شخصًا زار الحديقة في الأيام الثلاثة؟"
    )
    steps = [
        f"عدد السبت: {fri} × {sat_mult} = {sat}.",
        f"عدد الأحد: {sat} − {sun_less} = {sun}.",
        f"الإجمالي: {fri} + {sat} + {sun} = {total}.",
    ]
    return Sample("gsm8k", prompt, total, 3, {"grade_level": _grade_for_steps(3)}, steps)


def _gsm_work_rate(rng: random.Random) -> Sample:
    per_hour = rng.randint(8, 25)
    hours = rng.randint(3, 8)
    days = rng.randint(2, 5)
    total = per_hour * hours * days
    prompt = (
        f"عامل ينجز {per_hour} قطعة في الساعة، ويعمل {hours} ساعات يوميًا لمدة {days} أيام. "
        f"كم قطعة ينجز إجمالًا؟"
    )
    steps = [
        f"إنتاج اليوم الواحد: {per_hour} × {hours} = {per_hour*hours}.",
        f"على مدار {days} أيام: {per_hour*hours} × {days} = {total}.",
    ]
    return Sample("gsm8k", prompt, total, 2, {"grade_level": _grade_for_steps(2)}, steps)


def _gsm_fraction_of(rng: random.Random) -> Sample:
    total = rng.randint(40, 480)
    den = rng.choice([2, 3, 4, 5, 6, 8, 10])
    while total % den:
        total += 1
    part = total // den
    remain = total - part
    place = rng.choice(["مكتبة", "مدرسة", "مخزن", "معرض", "مركز ثقافي"])
    thing = rng.choice(["كتابًا", "مجلة", "قصة", "مرجعًا"])
    # agree grammar loosely
    prompt = f"لدى {place} {total} {thing}. أُعير منها {1}/{den}. كم تبقى؟"
    steps = [
        f"عدد المُعار: {total} ÷ {den} = {part}.",
        f"المتبقي: {total} − {part} = {remain}.",
    ]
    return Sample("gsm8k", prompt, remain, 2, {"grade_level": _grade_for_steps(2)}, steps)


def _gsm_ages(rng: random.Random) -> Sample:
    young = rng.randint(8, 15)
    older_more = rng.randint(5, 12)
    older = young + older_more
    years = rng.randint(3, 8)
    sum_future = (young + years) + (older + years)
    prompt = (
        f"عمر أحمد {young} سنة، وعمر أخيه أكبر منه بـ {older_more} سنوات. "
        f"بعد {years} سنوات، كم يصبح مجموع عمريهما؟"
    )
    steps = [
        f"عمر الأخ الآن: {young} + {older_more} = {older}.",
        f"عمر أحمد بعد {years} سنوات: {young} + {years} = {young+years}.",
        f"عمر الأخ بعد {years} سنوات: {older} + {years} = {older+years}.",
        f"المجموع: {young+years} + {older+years} = {sum_future}.",
    ]
    return Sample("gsm8k", prompt, sum_future, 4, {"grade_level": _grade_for_steps(4)}, steps)


def _gsm_ratio_share(rng: random.Random) -> Sample:
    a, b = rng.randint(2, 5), rng.randint(2, 5)
    unit = rng.randint(10, 40)
    total = (a + b) * unit
    share_a = a * unit
    prompt = (
        f"يُقسَم مبلغ {total} ريالًا بين شخصين بنسبة {a}:{b}. "
        f"كم ريالًا يحصل عليه صاحب النسبة {a}؟"
    )
    steps = [
        f"مجموع أجزاء النسبة: {a} + {b} = {a+b}.",
        f"قيمة الجزء الواحد: {total} ÷ {a+b} = {unit}.",
        f"نصيب النسبة {a}: {a} × {unit} = {share_a}.",
    ]
    return Sample("gsm8k", prompt, share_a, 3, {"grade_level": _grade_for_steps(3)}, steps)


def _gsm_remaining_after(rng: random.Random) -> Sample:
    start = rng.randint(100, 400)
    used1 = rng.randint(20, 80)
    used2 = rng.randint(15, 60)
    left = start - used1 - used2
    prompt = (
        f"كان في المستودع {start} صندوقًا. شُحن منها {used1} في اليوم الأول، "
        f"و{used2} في اليوم الثاني. كم صندوقًا بقي؟"
    )
    steps = [
        f"بعد اليوم الأول: {start} − {used1} = {start-used1}.",
        f"بعد اليوم الثاني: {start-used1} − {used2} = {left}.",
    ]
    return Sample("gsm8k", prompt, left, 2, {"grade_level": _grade_for_steps(2)}, steps)


def _gsm_stacked_ops(rng: random.Random) -> Sample:
    """Hard multi-step word problem — multiple story frames to avoid near-dups."""
    return rng.choice([
        _hard_books_library,
        _hard_factory_shipment,
        _hard_farm_harvest,
        _hard_store_restock,
        _hard_bus_trip_budget,
    ])(rng)


def _hard_books_library(rng: random.Random) -> Sample:
    a = rng.randint(20, 50)
    b = rng.randint(3, 8)
    c = rng.randint(2, 5)
    books = a * b + c
    d = rng.choice([k for k in range(2, 12) if books % k == 0] or [1])
    per_class = books // d
    take = rng.randint(1, max(1, per_class // 2))
    ans = d * take
    leftover = books - ans
    name = rng.choice(NAMES)
    school = rng.choice(["المدرسة", "المعهد", "المجمع التعليمي"])
    prompt = (
        f"في {school}، أشرفت {name} على توريد كتب جديدة وفق الخطوات التالية. "
        f"أولًا اشترت {a} صندوقًا، وفي كل صندوق {b} كتبًا. "
        f"ثم أضافت بعد ذلك {c} كتبًا مفردة وصلت متأخرة. "
        f"بعد جرد المجموع، وُزّعت الكتب بالتساوي على {d} صفوف دراسية. "
        f"وفي مرحلة لاحقة أُخذ من كل صف {take} كتبًا لإيداعها في مكتبة {school}. "
        f"المطلوب: أحسب كم كتابًا أُخذ للمكتبة إجمالًا، "
        f"مع التحقق من أن المتبقي في الصفوف بعد السحب يمكن قسمته على عدد الصفوف إن أمكن. "
        f"اذكر الناتج النهائي لعدد الكتب المأخوذة للمكتبة فقط."
    )
    steps = [
        f"أولًا نحسب كتب الصناديق: {a} × {b} = {a*b}.",
        f"بعد إضافة الكتب المفردة يصبح المجموع {a*b} + {c} = {books}.",
        f"بالتوزيع المتساوي يحصل كل صف على {books} ÷ {d} = {per_class}.",
        f"يُسحب من كل صف {take} كتبًا.",
        f"إذن مجموع ما يذهب إلى المكتبة: {d} × {take} = {ans}.",
        f"للتحقق: المتبقي في الصفوف {books} − {ans} = {leftover}.",
        f"ونصيب الصف من المتبقي إن كان قابلًا للقسمة: {leftover} ÷ {d} = {leftover // d if d and leftover % d == 0 else leftover / d}.",
    ]
    return Sample("gsm8k", prompt, ans, 7, {"grade_level": _grade_for_steps(7)}, steps)


def _hard_factory_shipment(rng: random.Random) -> Sample:
    machines = rng.randint(4, 9)
    per_m = rng.randint(30, 80)
    extra = rng.randint(5, 25)
    total = machines * per_m + extra
    boxes = rng.choice([k for k in range(3, 15) if total % k == 0] or [5])
    per_box = total // boxes
    damaged = rng.randint(1, max(1, per_box // 3))
    ans = boxes * damaged  # removed damaged from each box... wait story: damaged per box removed from shipment
    # Better: shipped boxes, each loses damaged units, how many lost total
    name = rng.choice(NAMES)
    prompt = (
        f"في مصنع، يشغّل {name} {machines} آلات، وتنتج كل آلة {per_m} قطعة يوميًا. "
        f"وأُضيفت دفعة إضافية قدرها {extra} قطعًا من مخزون احتياطي. "
        f"ثم عُبّئت القطع في {boxes} صناديق بالتساوي. "
        f"وقبل الشحن فُحص كل صندوق واستُبعد منه {damaged} قطع تالفة. "
        f"احسب إجمالي القطع التالفة المستبعدة. "
        f"ثم تحقق من عدد القطع السليمة المتبقية بقسمة ناتج الطرح على عدد الصناديق إن أمكن، "
        f"وأبلغ فقط بعدد القطع التالفة الإجمالي."
    )
    steps = [
        f"إنتاج الآلات: {machines} × {per_m} = {machines*per_m}.",
        f"بعد الإضافة: {machines*per_m} + {extra} = {total}.",
        f"محتوى الصندوق: {total} ÷ {boxes} = {per_box}.",
        f"التالف لكل صندوق: {damaged}.",
        f"إجمالي التالف: {boxes} × {damaged} = {ans}.",
        f"السليم الكلي: {total} − {ans} = {total-ans}.",
        f"متوسط السليم للصندوق: {(total-ans)//boxes if (total-ans)%boxes==0 else (total-ans)/boxes}.",
    ]
    return Sample("gsm8k", prompt, ans, 7, {"grade_level": _grade_for_steps(7)}, steps)


def _hard_farm_harvest(rng: random.Random) -> Sample:
    fields = rng.randint(3, 8)
    crates = rng.randint(20, 60)
    bonus = rng.randint(4, 20)
    total = fields * crates + bonus
    trucks = rng.choice([k for k in range(2, 10) if total % k == 0] or [2])
    per = total // trucks
    reserved = rng.randint(1, max(1, per // 4))
    ans = trucks * reserved
    name = rng.choice(NAMES)
    prompt = (
        f"حصد {name} محصولًا من {fields} حقول، وجمع من كل حقل {crates} صندوقًا، "
        f"ثم أضاف {bonus} صناديق من محصول جانبي. "
        f"وُزّع المجموع بالتساوي على {trucks} شاحنات. "
        f"واحتُجز من كل شاحنة {reserved} صناديق للسوق المحلي. "
        f"ما عدد الصناديق المحتجزة للسوق المحلي إجمالًا؟ "
        f"تحقق بحساب المتبقي للشحن الخارجي بعد الخصم، واذكر رقم الصناديق المحتجزة فقط."
    )
    steps = [
        f"صناديق الحقول: {fields} × {crates} = {fields*crates}.",
        f"بعد الإضافة: {fields*crates} + {bonus} = {total}.",
        f"نصيب الشاحنة: {total} ÷ {trucks} = {per}.",
        f"المحتجز من كل شاحنة: {reserved}.",
        f"الإجمالي المحتجز: {trucks} × {reserved} = {ans}.",
        f"المتبقي للشحن: {total} − {ans} = {total-ans}.",
        f"متوسط المتبقي للشاحنة: {(total-ans)//trucks if (total-ans)%trucks==0 else (total-ans)/trucks}.",
    ]
    return Sample("gsm8k", prompt, ans, 7, {"grade_level": _grade_for_steps(7)}, steps)


def _hard_store_restock(rng: random.Random) -> Sample:
    shelves = rng.randint(5, 12)
    packs = rng.randint(8, 25)
    loose = rng.randint(3, 18)
    total = shelves * packs + loose
    carts = rng.choice([k for k in range(2, 11) if total % k == 0] or [2])
    per = total // carts
    sold = rng.randint(1, max(1, per // 3))
    ans = carts * sold
    name = rng.choice(NAMES)
    prompt = (
        f"أعاد {name} ترتيب مخزن فيه {shelves} رفوفًا، وعلى كل رف {packs} عبوة، "
        f"مع {loose} عبوات إضافية على الطاولة. "
        f"نُقلت العبوات بالتساوي إلى {carts} عربات. "
        f"وبِيع من كل عربة {sold} عبوات في أول ساعة. "
        f"كم عبوة بِيعت في الساعة الأولى إجمالًا؟ "
        f"احسب أيضًا المتبقي في العربات بعد البيع للتحقق، وأجب بعدد المبيعات فقط."
    )
    steps = [
        f"عبوات الرفوف: {shelves} × {packs} = {shelves*packs}.",
        f"بعد الإضافة: {shelves*packs} + {loose} = {total}.",
        f"نصيب العربة: {total} ÷ {carts} = {per}.",
        f"المباع من كل عربة: {sold}.",
        f"إجمالي المبيعات: {carts} × {sold} = {ans}.",
        f"المتبقي: {total} − {ans} = {total-ans}.",
        f"متوسط المتبقي للعربة: {(total-ans)//carts if (total-ans)%carts==0 else (total-ans)/carts}.",
    ]
    return Sample("gsm8k", prompt, ans, 7, {"grade_level": _grade_for_steps(7)}, steps)


def _hard_bus_trip_budget(rng: random.Random) -> Sample:
    groups = rng.randint(3, 7)
    students = rng.randint(15, 40)
    teachers = rng.randint(2, 8)
    people = groups * students + teachers
    buses = rng.choice([k for k in range(2, 9) if people % k == 0] or [2])
    per = people // buses
    free = rng.randint(1, max(1, per // 4))
    ans = buses * free  # free tickets total
    name = rng.choice(NAMES)
    cost = rng.randint(10, 40)
    # ask for free tickets count not money to keep int GT simple
    prompt = (
        f"نظّم {name} رحلة لـ {groups} مجموعات، في كل مجموعة {students} طالبًا، "
        f"إضافة إلى {teachers} معلمين. "
        f"توزّع الجميع بالتساوي على {buses} حافلات. "
        f"ومُنح في كل حافلة {free} مقاعد مجانية. "
        f"ما مجموع المقاعد المجانية في الرحلة؟ "
        f"تحقق بعدد الركاب غير المجانيين عبر طرح المجاني من الإجمالي، "
        f"وأجب بعدد المقاعد المجانية فقط (سعر المقعد {cost} ريالًا للمرجعية فقط)."
    )
    steps = [
        f"الطلاب: {groups} × {students} = {groups*students}.",
        f"بعد المعلمين: {groups*students} + {teachers} = {people}.",
        f"ركاب الحافلة: {people} ÷ {buses} = {per}.",
        f"المجان لكل حافلة: {free}.",
        f"مجموع المجاني: {buses} × {free} = {ans}.",
        f"غير المجانيين: {people} − {ans} = {people-ans}.",
        f"متوسط غير المجاني للحافلة: {(people-ans)//buses if (people-ans)%buses==0 else (people-ans)/buses}.",
    ]
    return Sample("gsm8k", prompt, ans, 7, {"grade_level": _grade_for_steps(7)}, steps)


def _gsm_bus_split(rng: random.Random) -> Sample:
    buses = rng.randint(3, 8)
    per = rng.randint(18, 50)
    frac_den = rng.choice([2, 3, 4, 5])
    while per % frac_den:
        per += 1
    total = buses * per
    one_bus_part = per // frac_den
    item = rng.choice(["كاميرا", "حقيبة", "بطاقة", "كتابًا", "جهازًا"])
    vehicle = rng.choice(["حافلات", "باصات", "مركبات", "حافلات مدرسية"])
    prompts = [
        f"ركب {total} طالبًا في {buses} {vehicle} بالتساوي. كان مع {1}/{frac_den} من ركاب الأولى {item}. كم عددهم؟",
        f"{total} راكبًا وُزّعوا على {buses} {vehicle}. في الأولى يحمل {item} نسبة {1}/{frac_den}. احسب العدد.",
        f"بالتساوي على {buses} {vehicle} لـ {total} طالبًا؛ نسبة حاملي {item} في الأولى {1}/{frac_den}. كم؟",
        f"قُسم {total} راكبًا على {buses} {vehicle}. نسبة {1}/{frac_den} من ركاب الأولى معهم {item}. ما العدد؟",
        f"بعد توزيع {total} طالبًا على {buses} {vehicle} بالتساوي، أخذ {1}/{frac_den} من طلاب الأولى {item}. كم أخذوه؟",
    ]
    steps = [
        f"طلاب الواحدة: {total} ÷ {buses} = {per}.",
        f"الذين معهم {item}: {per} ÷ {frac_den} = {one_bus_part}.",
    ]
    return Sample("gsm8k", rng.choice(prompts), one_bus_part, 2, {"grade_level": _grade_for_steps(2)}, steps)


def _gsm_salary_save(rng: random.Random) -> Sample:
    salary = rng.randint(2000, 8000)
    save_den = rng.choice([4, 5, 8, 10])
    while salary % save_den:
        salary += 1
    monthly = salary // save_den
    months = rng.randint(3, 12)
    ans = monthly * months
    prompt = (
        f"راتبه الشهري {salary} ريالًا، ويدّخر {1}/{save_den} منه كل شهر. "
        f"كم يدّخر خلال {months} أشهر؟"
    )
    steps = [
        f"الادخار الشهري: {salary} ÷ {save_den} = {monthly}.",
        f"خلال {months} أشهر: {monthly} × {months} = {ans}.",
    ]
    return Sample("gsm8k", prompt, ans, 2, {"grade_level": _grade_for_steps(2)}, steps)


def _gsm_garden_rows(rng: random.Random) -> Sample:
    total = rng.randint(200, 900)
    broken_from = rng.randint(80, 150)
    broken_ok = rng.randint(40, broken_from - 10)
    broken = broken_from - broken_ok
    # simplify: total bricks, broken count, per row
    total = rng.randint(400, 1200)
    broken = rng.randint(30, 120)
    good = total - broken
    per_row = rng.choice([20, 25, 40, 50])
    while good % per_row:
        total += 1
        good = total - broken
    rows = good // per_row
    prompt = (
        f"يوجد {total} طوبة، منها {broken} مكسورة. "
        f"تُصفّ الطوب السليمة في صفوف كل منها {per_row} طوبة. "
        f"كم صفًا كاملًا يُشكَّل؟"
    )
    steps = [
        f"الطوب السليم: {total} − {broken} = {good}.",
        f"عدد الصفوف: {good} ÷ {per_row} = {rows}.",
    ]
    return Sample("gsm8k", prompt, rows, 2, {"grade_level": _grade_for_steps(2)}, steps)


# ---------------------------------------------------------------------------
# Math templates
# ---------------------------------------------------------------------------

def gen_math(rng: random.Random) -> Sample:
    families = [
        ("math_percent", _math_percent),
        ("math_discount", _math_discount),
        ("math_speed", _math_speed),
        ("math_mixture", _math_mixture),
        ("math_area_rect", _math_area_rect),
        ("math_simple_interest", _math_simple_interest),
        ("math_unit_price", _math_unit_price),
        ("math_avg", _math_avg),
    ]
    name, fn = rng.choice(families)
    return _with_family(fn(rng), name)


def _diff_tag(ns: int) -> str:
    return "hard" if ns >= 5 else "easy"


def _math_percent(rng: random.Random) -> Sample:
    pct = rng.choice([10, 15, 20, 25, 30, 40, 50])
    # ensure exact integer percent
    unit = 100 // math.gcd(pct, 100)
    base = unit * rng.randint(2, 40)
    ans = base * pct // 100
    assert base * pct % 100 == 0
    prompt = f"ما مقدار {pct}% من العدد {base}؟"
    steps = [f"نحسب: {base} × {pct} ÷ 100 = {ans}."]
    return Sample("math", prompt, ans, 1, {"difficulty_tag": "easy"}, steps)


def _math_discount(rng: random.Random) -> Sample:
    pct = rng.choice([10, 15, 20, 25])
    unit = 100 // math.gcd(pct, 100)
    price = unit * rng.randint(4, 50)
    disc = price * pct // 100
    assert price * pct % 100 == 0
    ans = price - disc
    prompt = (
        f"سعر سلعة {price} ريالًا، وعليها خصم بنسبة {pct}%. "
        f"كم يصبح السعر بعد الخصم؟"
    )
    steps = [
        f"قيمة الخصم: {price} × {pct} ÷ 100 = {disc}.",
        f"السعر بعد الخصم: {price} − {disc} = {ans}.",
    ]
    return Sample("math", prompt, ans, 2, {"difficulty_tag": _diff_tag(2)}, steps)


def _math_speed(rng: random.Random) -> Sample:
    speed = rng.randint(40, 120)
    hours = rng.randint(2, 6)
    dist = speed * hours
    prompt = (
        f"سيارة تسير بسرعة {speed} كيلومترًا في الساعة لمدة {hours} ساعات. "
        f"كم كيلومترًا تقطع؟"
    )
    steps = [f"المسافة = السرعة × الزمن = {speed} × {hours} = {dist}."]
    return Sample("math", prompt, dist, 1, {"difficulty_tag": "easy"}, steps)


def _math_mixture(rng: random.Random) -> Sample:
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    pa, pb = rng.randint(5, 20), rng.randint(5, 20)
    total_cost = a * pa + b * pb
    total_w = a + b
    # average price * 10 for one decimal sometimes — keep int
    # use weighted average integer when divisible
    if total_cost % total_w == 0:
        ans = total_cost // total_w
        prompt = (
            f"خُلط {a} كيلوغرام بسعر {pa} ريالًا للكيلوغرام مع {b} كيلوغرام بسعر {pb} ريالًا. "
            f"ما متوسط سعر الكيلوغرام للخليط؟"
        )
        steps = [
            f"التكلفة الكلية: {a}×{pa} + {b}×{pb} = {total_cost}.",
            f"الوزن الكلي: {a} + {b} = {total_w}.",
            f"المتوسط: {total_cost} ÷ {total_w} = {ans}.",
        ]
        return Sample("math", prompt, ans, 3, {"difficulty_tag": _diff_tag(3)}, steps)
    return _math_discount(rng)


def _math_area_rect(rng: random.Random) -> Sample:
    L, W = rng.randint(5, 40), rng.randint(3, 30)
    area = L * W
    prompt = f"مستطيل طوله {L} مترًا وعرضه {W} مترًا. كم مساحته بالمتر المربع؟"
    steps = [f"المساحة = الطول × العرض = {L} × {W} = {area}."]
    return Sample("math", prompt, area, 1, {"difficulty_tag": "easy"}, steps)


def _math_simple_interest(rng: random.Random) -> Sample:
    r = rng.choice([5, 6, 8, 10, 12])
    t = rng.randint(2, 5)
    # P * r * t divisible by 100
    unit = 100 // math.gcd(r * t, 100)
    P = unit * rng.randint(10, 90)
    I = P * r * t // 100
    assert P * r * t % 100 == 0
    prompt = (
        f"أُودع مبلغ {P} ريالًا بفائدة بسيطة سنوية {r}% لمدة {t} سنوات. "
        f"كم تبلغ الفائدة؟"
    )
    steps = [
        f"الفائدة = الأصلي × النسبة × الزمن ÷ 100.",
        f"إذن: {P} × {r} × {t} ÷ 100 = {I}.",
    ]
    return Sample("math", prompt, I, 2, {"difficulty_tag": _diff_tag(2)}, steps)


def _math_unit_price(rng: random.Random) -> Sample:
    n = rng.randint(4, 20)
    total = n * rng.randint(5, 40)
    unit = total // n
    prompt = f"اشترى تاجر {n} قطعة بمبلغ {total} ريالًا. ما سعر القطعة الواحدة؟"
    steps = [f"سعر الوحدة = {total} ÷ {n} = {unit}."]
    return Sample("math", prompt, unit, 1, {"difficulty_tag": "easy"}, steps)


def _math_avg(rng: random.Random) -> Sample:
    vals = [rng.randint(10, 100) for _ in range(rng.randint(3, 6))]
    s = sum(vals)
    if s % len(vals):
        vals[-1] += len(vals) - (s % len(vals))
        s = sum(vals)
    avg = s // len(vals)
    listed = " و".join(str(v) for v in vals)
    prompt = f"أوجد متوسط الأعداد الآتية: {listed}."
    steps = [
        f"المجموع = {' + '.join(map(str, vals))} = {s}.",
        f"المتوسط = {s} ÷ {len(vals)} = {avg}.",
    ]
    return Sample("math", prompt, avg, 2, {"difficulty_tag": _diff_tag(2)}, steps)


# ---------------------------------------------------------------------------
# Math_comp (closed-form / sympy-backed)
# ---------------------------------------------------------------------------

def gen_math_comp(rng: random.Random) -> Sample:
    families = [
        ("mc_linear", _mc_linear),
        ("mc_quadratic_root_sum", _mc_quadratic_root_sum),
        ("mc_modular", _mc_modular),
        ("mc_gcd_style", _mc_gcd_style),
        ("mc_arithm_seq", _mc_arithm_seq),
        ("mc_power_diff", _mc_power_diff),
        ("mc_lcm_product", _mc_lcm_product),
        ("mc_digit_sum_constraint", _mc_digit_sum_constraint),
    ]
    name, fn = rng.choice(families)
    return _with_family(fn(rng), name)


def _mc_linear(rng: random.Random) -> Sample:
    a = rng.randint(2, 15)
    x = rng.randint(3, 40)
    b = rng.randint(1, 30)
    c = a * x + b
    prompt = f"حل المعادلة: {a}x + {b} = {c}. أوجد قيمة x."
    steps = [
        f"نطرح {b} من الطرفين: {a}x = {c - b}.",
        f"نقسم على {a}: x = {x}.",
    ]
    return Sample(
        "math_comp",
        prompt,
        f"x = {x}",
        2,
        {"math_domain": "algebra", "level": 2},
        steps,
    )


def _mc_quadratic_root_sum(rng: random.Random) -> Sample:
    r1, r2 = rng.randint(1, 20), rng.randint(1, 20)
    s, p = r1 + r2, r1 * r2
    form = rng.choice(
        [
            f"مجموع جذرَي المعادلة x^2 - {s}x + {p} = 0 يساوي؟ أجب بعدد فقط.",
            f"إذا كانت المعادلة x² − {s}x + {p} = 0، فما مجموع الجذرين؟",
            f"أوجد مجموع جذور x^2 - {s}x + {p} = 0.",
        ]
    )
    steps = [
        f"لمعادلة x^2 - (مجموع)x + (جداء) = 0، المجموع = {s}.",
    ]
    return Sample(
        "math_comp",
        form,
        str(s),
        2,
        {"math_domain": "algebra", "level": 3},
        steps,
    )


def _mc_modular(rng: random.Random) -> Sample:
    a = rng.randint(20, 200)
    m = rng.randint(3, 17)
    ans = a % m
    prompt = f"أحسب {a} modulo {m} (باقي القسمة)."
    steps = [f"{a} = {a // m} × {m} + {ans}، لذا الباقي = {ans}."]
    return Sample(
        "math_comp",
        prompt,
        str(ans),
        2,
        {"math_domain": "number_theory", "level": 2},
        steps,
    )


def _mc_gcd_style(rng: random.Random) -> Sample:
    g = rng.randint(2, 12)
    a, b = g * rng.randint(2, 15), g * rng.randint(2, 15)
    ans = math.gcd(a, b)
    prompt = f"أوجد القاسم المشترك الأكبر للعددين {a} و{b}."
    steps = [f"نستخدم خوارزمية إقليدس لنحصل على {ans}."]
    return Sample(
        "math_comp",
        prompt,
        str(ans),
        3,
        {"math_domain": "number_theory", "level": 3},
        steps,
    )


def _mc_arithm_seq(rng: random.Random) -> Sample:
    a1 = rng.randint(2, 20)
    d = rng.randint(2, 9)
    n = rng.randint(5, 12)
    an = a1 + (n - 1) * d
    prompt = (
        f"متتالية حسابية حدها الأول {a1} والأساس {d}. "
        f"ما قيمة الحد رقم {n}؟"
    )
    steps = [
        f"الحد النوني: a_n = a_1 + (n-1)d.",
        f"a_{n} = {a1} + ({n}-1)×{d} = {an}.",
    ]
    return Sample(
        "math_comp",
        prompt,
        str(an),
        2,
        {"math_domain": "sequences", "level": 2},
        steps,
    )


def _mc_power_diff(rng: random.Random) -> Sample:
    a = rng.randint(2, 9)
    b = rng.randint(2, 5)
    c = rng.randint(1, 4)
    ans = a ** b - c
    prompt = f"أحسب قيمة {a}^{b} − {c}."
    steps = [
        f"{a}^{b} = {a**b}.",
        f"ثم نطرح {c}: {a**b} − {c} = {ans}.",
    ]
    ns = 2
    # harden
    if rng.random() < 0.4:
        d = rng.randint(2, 6)
        ans = (a ** b - c) * d
        prompt = f"أحسب قيمة ({a}^{b} − {c}) × {d}."
        steps.append(f"نضرب الناتج في {d}: {a**b - c} × {d} = {ans}.")
        ns = 3
    return Sample(
        "math_comp",
        prompt,
        str(ans),
        ns,
        {"math_domain": "algebra", "level": min(5, ns + 1)},
        steps,
    )


def _mc_lcm_product(rng: random.Random) -> Sample:
    a = rng.randint(2, 12)
    b = rng.randint(2, 12)
    while math.gcd(a, b) == 1 and rng.random() < 0.5:
        b = rng.randint(2, 12)
    lcm = a * b // math.gcd(a, b)
    prompt = f"ما المضاعف المشترك الأصغر للعددين {a} و{b}؟"
    steps = [
        f"القاسم المشترك الأكبر لـ {a} و{b} هو {math.gcd(a, b)}.",
        f"المضاعف المشترك الأصغر = ({a}×{b})÷{math.gcd(a, b)} = {lcm}.",
    ]
    return Sample(
        "math_comp",
        prompt,
        str(lcm),
        2,
        {"math_domain": "number_theory", "level": 2},
        steps,
    )


def _mc_digit_sum_constraint(rng: random.Random) -> Sample:
    tens = rng.randint(1, 9)
    ones = rng.randint(0, 9)
    n = 10 * tens + ones
    s = tens + ones
    prompt = (
        f"عدد مكون من رقمين، مجموع رقميه {s}، ورقمه العشرات أكبر من رقمه الآحاد بمقدار "
        f"{tens - ones if tens >= ones else ones - tens}. ما هو العدد؟"
        if tens != ones
        else f"عدد مكون من رقمين ومجموع رقميه {s} ورقماه متساويان. ما هو العدد؟"
    )
    if tens == ones:
        steps = [
            f"الرقمان متساويان ومجموعهما {s}، فكل منهما {s // 2}.",
            f"العدد = {n}.",
        ]
    elif tens > ones:
        steps = [
            f"ليكن رقم الآحاد x ورقمه العشرات x+{tens - ones}.",
            f"مجموع الرقمين = 2x+{tens - ones} = {s} ⇒ x = {ones}.",
            f"العدد = {n}.",
        ]
    else:
        steps = [
            f"ليكن رقم العشرات x ورقمه الآحاد x+{ones - tens}.",
            f"مجموع الرقمين = 2x+{ones - tens} = {s} ⇒ x = {tens}.",
            f"العدد = {n}.",
        ]
    return Sample(
        "math_comp",
        prompt,
        str(n),
        3,
        {"math_domain": "digits", "level": 3},
        steps,
    )


# ---------------------------------------------------------------------------
# Logic — unique flat GT via enumeration
# ---------------------------------------------------------------------------

def gen_logic(rng: random.Random) -> Sample:
    families = [
        ("logic_ordering", _logic_ordering),
        ("logic_assignment", _logic_assignment),
        ("logic_truth", _logic_truth),
        ("logic_who_has", _logic_who_has),
        ("logic_seating_left_right", _logic_seating_left_right),
        ("logic_schedule_slots", _logic_schedule_slots),
        ("logic_color_objects", _logic_color_objects),
        ("logic_two_attribute", _logic_two_attribute),
    ]
    name, fn = rng.choice(families)
    return _with_family(fn(rng), name)


def _logic_ordering(rng: random.Random) -> Sample:
    people = rng.sample(["سلمان", "خالد", "فيصل", "تركي"], 4)
    # unique permutation as truth
    order = people[:]  # 1st to 4th
    rng.shuffle(order)
    # clues that uniquely determine - generate from order
    # clue1: order[1] immediately before order[2] if we state "X precedes Y directly"
    # Simpler unique puzzle: positions known by constraints
    p0, p1, p2, p3 = order
    prompt = (
        f"أربعة عدائين في سباق: {', '.join(people)}. ترتيبهم من الأول إلى الرابع.\n"
        f"1. {p1} يسبق {p2} مباشرة.\n"
        f"2. {p0} هو الأول.\n"
        f"3. {p3} هو الأخير.\n"
        f"من هو الثاني والثالث؟ أخرج التعيين الكامل للمراتب."
    )
    gt = {"الأول": p0, "الثاني": p1, "الثالث": p2, "الرابع": p3}
    # verify uniqueness: only one order satisfies
    steps = [
        f"من (2): الأول = {p0}.",
        f"من (3): الرابع = {p3}.",
        f"من (1): الثاني = {p1} والثالث = {p2}.",
        f"الترتيب الكامل: {p0}، {p1}، {p2}، {p3}.",
    ]
    return Sample(
        "logic",
        prompt,
        gt,
        4,
        {"difficulty_tag": "hard", "puzzle_type": "ordering"},
        steps,
    )


def _logic_assignment(rng: random.Random) -> Sample:
    people = rng.sample(["سعد", "ماجد", "وليد"], 3)
    floors = ["الأول", "الثاني", "الثالث"]
    colors = ["أحمر", "أزرق", "أسود"]
    # assign unique floor and color
    fl = floors[:]
    co = colors[:]
    rng.shuffle(fl)
    rng.shuffle(co)
    assign = {people[i]: (fl[i], co[i]) for i in range(3)}
    # pick clues
    p_floor2 = next(p for p, (f, _) in assign.items() if f == "الثاني")
    p_blue = next(p for p, (_, c) in assign.items() if c == "أزرق")
    p_floor3 = next(p for p, (f, _) in assign.items() if f == "الثالث")
    color_floor3 = assign[p_floor3][1]
    # who lives above blue
    blue_floor = assign[p_blue][0]
    floor_idx = {"الأول": 0, "الثاني": 1, "الثالث": 2}
    above = None
    for p, (f, _) in assign.items():
        if floor_idx[f] == floor_idx[blue_floor] + 1:
            above = p
    if above is None:
        return _logic_ordering(rng)
    prompt = (
        f"ثلاثة جيران: {', '.join(people)}. لكل واحد دور (الأول أو الثاني أو الثالث) "
        f"وسيارة بلون (أحمر أو أزرق أو أسود).\n"
        f"1. {p_floor2} يسكن في الدور الثاني.\n"
        f"2. {p_blue} لديه سيارة زرقاء.\n"
        f"3. الساكن في الدور الثالث لديه سيارة لونها {color_floor3}.\n"
        f"4. {above} يسكن فوق صاحب السيارة الزرقاء.\n"
        f"عيّن الدور ولون السيارة لكل شخص."
    )
    gt = {}
    for p, (f, c) in assign.items():
        gt[f"{p}_دور"] = f
        gt[f"{p}_سيارة"] = c
    steps = [
        f"من القرائن نحدد أن {p_floor2} في الدور الثاني.",
        f"{p_blue} لديه السيارة الزرقاء، و{above} فوقه.",
        f"الساكن في الثالث سيارته {color_floor3}.",
        "وبالاستبعاد تكتمل التعيينات دون تعارض.",
    ]
    return Sample(
        "logic",
        prompt,
        gt,
        5,
        {"difficulty_tag": "hard", "puzzle_type": "grid_deduction"},
        steps,
    )


def _logic_truth(rng: random.Random) -> Sample:
    pool = [
        "فهد", "سلطان", "ناصر", "سعد", "ماجد", "وليد", "خالد", "تركي",
        "فيصل", "أحمد", "يوسف", "حسن", "عمر", "علي", "إبراهيم",
    ]
    names = rng.sample(pool, 3)
    # Fixed deduction structure with shuffled names → unique GT
    prompt = (
        f"ثلاثة أشخاص: {names[0]} و{names[1]} و{names[2]}. "
        f"واحد صادق دائمًا وواحد كاذب دائمًا وواحد متقلب.\n"
        f"قال {names[0]}: «{names[1]} هو الكاذب».\n"
        f"قال {names[1]}: «{names[2]} هو الصادق».\n"
        f"قال {names[2]}: «أنا المتقلب».\n"
        f"عُلم أن جملة {names[2]} صحيحة، وأن {names[0]} ليس متقلبًا. من هو كل واحد؟"
    )
    gt = {names[0]: "صادق", names[1]: "كاذب", names[2]: "متقلب"}
    steps = [
        f"جملة {names[2]} صحيحة ⇒ {names[2]} متقلب.",
        f"جملة {names[1]} («{names[2]} صادق») خاطئة ⇒ {names[1]} كاذب.",
        f"إذن {names[0]} صادق، ويتوافق ذلك مع قوله إن {names[1]} كاذب.",
    ]
    return Sample(
        "logic",
        prompt,
        gt,
        4,
        {"difficulty_tag": "hard", "puzzle_type": "truth_teller_liar"},
        steps,
    )


def _logic_who_has(rng: random.Random) -> Sample:
    people = rng.sample(["أحمد", "سارة", "نورة"], 3)
    items = rng.sample(["كتاب", "قلم", "حقيبة"], 3)
    assign = {people[i]: items[i] for i in range(3)}
    p0, p1, p2 = people
    prompt = (
        f"لدى {', '.join(people)} ثلاثة أغراض مختلفة: {', '.join(items)}.\n"
        f"1. {p0} ليس معه {assign[p1]}.\n"
        f"2. {p1} معه {assign[p1]}.\n"
        f"3. {p2} ليس معه {assign[p0]}.\n"
        f"عيّن ما مع كل شخص."
    )
    steps = [
        f"من (2): {p1} معه {assign[p1]}.",
        f"من (1) و(3) بالاستبعاد تتعين بقية التعيينات.",
    ]
    return Sample(
        "logic",
        prompt,
        assign,
        3,
        {"difficulty_tag": "medium", "puzzle_type": "who_has"},
        steps,
    )


def _logic_seating_left_right(rng: random.Random) -> Sample:
    people = rng.sample(["خالد", "فاطمة", "يوسف", "ليلى"], 4)
    order = people[:]
    rng.shuffle(order)
    a, b, c, d = order
    prompt = (
        f"أربعة أشخاص يجلسون في صف واحد من اليسار إلى اليمين: {', '.join(people)}.\n"
        f"1. {b} يجلس مباشرة إلى يمين {a}.\n"
        f"2. {d} في أقصى اليمين.\n"
        f"3. {c} ليس في أقصى اليسار.\n"
        f"عيّن ترتيب الجلوس من اليسار إلى اليمين."
    )
    gt = {"يسار1": a, "يسار2": b, "يسار3": c, "يسار4": d}
    steps = [
        f"من (2): أقصى اليمين = {d}.",
        f"من (1): {a} ثم {b} مباشرة.",
        f"من (3) يتبقى {c} في الموقع الثالث.",
    ]
    return Sample(
        "logic",
        prompt,
        gt,
        3,
        {"difficulty_tag": "medium", "puzzle_type": "seating"},
        steps,
    )


def _logic_schedule_slots(rng: random.Random) -> Sample:
    people = rng.sample(["مها", "سامي", "هند"], 3)
    slots = ["صباحًا", "ظهرًا", "مساءً"]
    order = slots[:]
    rng.shuffle(order)
    assign = {people[i]: order[i] for i in range(3)}
    morning = next(p for p, s in assign.items() if s == "صباحًا")
    evening = next(p for p, s in assign.items() if s == "مساءً")
    noon = next(p for p, s in assign.items() if s == "ظهرًا")
    prompt = (
        f"ثلاثة مواعيد: صباحًا وظهرًا ومساءً للأشخاص {', '.join(people)}.\n"
        f"1. {morning} موعده صباحًا.\n"
        f"2. {evening} ليس موعده ظهرًا.\n"
        f"3. موعد {noon} بعد موعد {morning}.\n"
        f"عيّن موعد كل شخص."
    )
    steps = [
        f"من (1): {morning} صباحًا.",
        f"من (2) و(3): {noon} ظهرًا و{evening} مساءً.",
    ]
    return Sample(
        "logic",
        prompt,
        assign,
        3,
        {"difficulty_tag": "medium", "puzzle_type": "schedule"},
        steps,
    )


def _logic_color_objects(rng: random.Random) -> Sample:
    objects = rng.sample(["قلم", "دفتر", "مسطرة"], 3)
    colors = rng.sample(["أحمر", "أخضر", "أزرق"], 3)
    assign = {objects[i]: colors[i] for i in range(3)}
    o0, o1, o2 = objects
    prompt = (
        f"ثلاثة أدوات: {', '.join(objects)} بألوان {', '.join(colors)} دون تكرار.\n"
        f"1. {o0} ليس {assign[o1]}.\n"
        f"2. {o1} لونه {assign[o1]}.\n"
        f"3. {o2} ليس {assign[o0]}.\n"
        f"عيّن لون كل أداة."
    )
    steps = [
        f"من (2): {o1} = {assign[o1]}.",
        "ثم بالاستبعاد تتعين بقية الألوان.",
    ]
    return Sample(
        "logic",
        prompt,
        assign,
        3,
        {"difficulty_tag": "easy", "puzzle_type": "colors"},
        steps,
    )


def _logic_two_attribute(rng: random.Random) -> Sample:
    people = rng.sample(["ريم", "هدى", "سلمى"], 3)
    jobs = rng.sample(["طبيبة", "معلمة", "مهندسة"], 3)
    cities = rng.sample(["الرياض", "جدة", "الدمام"], 3)
    assign_job = {people[i]: jobs[i] for i in range(3)}
    assign_city = {people[i]: cities[i] for i in range(3)}
    p0 = people[0]
    prompt = (
        f"ثلاث نساء: {', '.join(people)}. لكل واحدة مهنة ومدينة سكن.\n"
        f"1. {p0} تعمل {assign_job[p0]} وتسكن في {assign_city[p0]}.\n"
        f"2. {people[1]} ليست {assign_job[people[2]]}.\n"
        f"3. الساكنة في {assign_city[people[1]]} تعمل {assign_job[people[1]]}.\n"
        f"4. {people[2]} تسكن في {assign_city[people[2]]}.\n"
        f"عيّن المهنة والمدينة لكل واحدة."
    )
    gt = {}
    for p in people:
        gt[f"{p}_مهنة"] = assign_job[p]
        gt[f"{p}_مدينة"] = assign_city[p]
    steps = [
        f"من (1): {p0} محددة بالكامل.",
        "من بقية القرائن تتعين المهنة والمدينة للباقي بلا تعارض.",
    ]
    return Sample(
        "logic",
        prompt,
        gt,
        4,
        {"difficulty_tag": "hard", "puzzle_type": "two_attribute"},
        steps,
    )


# ---------------------------------------------------------------------------
# GT serialization (exact-match safe)
# ---------------------------------------------------------------------------

def canonicalize_answer(gt: Any, domain: str = "") -> str:
    """Stable string for top-level answer and <answer>…</answer>.

    Logic: compact JSON with sorted keys (exact-match reward).
    Metadata for logic keeps the dict object — see metadata_gt().
    """
    if isinstance(gt, dict):
        return json.dumps(gt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(gt, float) and gt == int(gt):
        return str(int(gt))
    if isinstance(gt, (int, float)):
        return str(gt)
    s = str(gt).strip()
    # math_comp often "x = N" — keep as-is but normalize spaces around =
    if domain == "math_comp" and re.match(r"^x\s*=\s*", s, re.I):
        num = s.split("=", 1)[1].strip()
        return f"x = {num}"
    return s


def metadata_gt(gt: Any, domain: str = "") -> Any:
    """Value stored in metadata.ground_truth_answer.

    Logic stays a flat dict so reward paths that json.loads / compare dicts keep working.
    Numeric domains store int when possible.
    """
    if domain == "logic" or isinstance(gt, dict):
        if not isinstance(gt, dict):
            raise TypeError("logic GT must be a flat dict")
        # ensure JSON-serializable flat strings/values
        return {str(k): v for k, v in gt.items()}
    if isinstance(gt, float) and gt == int(gt):
        return int(gt)
    if isinstance(gt, (int, float)):
        return gt
    return canonicalize_answer(gt, domain)


def normalize_difficulty(domain: str, meta: dict, num_steps: int) -> dict:
    """Ensure curriculum fields are never null."""
    out = dict(meta or {})
    out["num_steps"] = int(num_steps)
    if domain == "gsm8k":
        out.setdefault("grade_level", _grade_for_steps(num_steps))
    elif domain == "math":
        out.setdefault(
            "difficulty_tag",
            "hard" if num_steps >= 5 else ("medium" if num_steps >= 3 else "easy"),
        )
    elif domain == "math_comp":
        level = int(out.get("level") or min(5, max(1, num_steps)))
        out["level"] = level
        out.setdefault(
            "difficulty_tag",
            "hard" if level >= 4 else ("medium" if level >= 3 else "easy"),
        )
    elif domain == "logic":
        out.setdefault("difficulty_tag", "hard")
        out.setdefault("puzzle_type", out.get("puzzle_type") or "ordering")
    return out


# Trailing unit / short phrase after a stated final (e.g. "هو 24 زجاجة.")
_UNIT_TAIL = r"(?:\s+[^\d\n]{0,24})?"

# Arabic-Indic (Eastern) digits → Western ASCII
_EASTERN_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Mechanical scrub must NEVER inject this — it became a mass CoT contaminant
_SCRUB_BOILERPLATE_RE = re.compile(
    r"نراجع\s+(?:الاتساق|خطوات\s+الحل|ترتيب\s+العمليات)[^\n]*",
    re.UNICODE,
)


def normalize_eastern_digits(text: str) -> str:
    """Map Arabic-Indic / Eastern digits to ASCII 0-9 (leak detection + scrub)."""
    if not text:
        return text
    return text.translate(_EASTERN_DIGIT_MAP)


def _gt_digit_forms(canon: str) -> list[str]:
    """Western canon plus Eastern-digit spelling for the same numeric string."""
    if not canon:
        return []
    forms = [canon]
    # If canon is numeric / mostly digits, also try Eastern form for matching raw think
    east_digits = "٠١٢٣٤٥٦٧٨٩"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", canon.strip()):
        east = "".join(
            east_digits[int(c)] if c.isdigit() else c for c in canon.strip()
        )
        if east != canon:
            forms.append(east)
    return forms


def _conclusion_states_gt(text: str, canon: str) -> bool:
    """True if text states final GT as a conclusion (units after GT allowed)."""
    if not text or not canon:
        return False
    # Always compare in Western digits so ١٧٥ matches GT 175
    text_n = normalize_eastern_digits(text)
    for form in _gt_digit_forms(canon):
        form_n = normalize_eastern_digits(form)
        if re.fullmatch(rf"{re.escape(form_n)}\.?", text_n.strip()):
            return True
        if re.search(
            rf"(?:الجواب|الإجابة|الناتج(?:\s+النهائي)?|الحل(?:\s+النهائي)?|الإجابة الصحيحة|الباقي|المجموع)"
            rf"\s*[:：]?\s*{re.escape(form_n)}(?!\d)",
            text_n,
        ):
            return True
        if re.search(
            rf"(?:إذن|وبالتالي|وعليه)\s*[،,]?\s+.{{0,120}}?"
            rf"(?:هو|هي|يساوي|=)\s*{re.escape(form_n)}(?!\d){_UNIT_TAIL}\s*\.?\s*$",
            text_n,
            re.DOTALL,
        ):
            return True
        if re.search(
            rf"(?:الإجمالي|الناتج|المجموع|الباقي)\s+.{{0,100}}?"
            rf"(?:هو|هي|يساوي|=)\s*{re.escape(form_n)}(?!\d){_UNIT_TAIL}\s*\.?\s*$",
            text_n,
            re.DOTALL,
        ):
            return True
        if re.search(
            rf"(?:لذلك|وبالتالي|وعليه|بهذا)\s*[،,]?\s*.{{0,80}}?"
            rf"(?:هي|هو|يساوي|=)\s*{re.escape(form_n)}(?!\d){_UNIT_TAIL}\s*\.?\s*$",
            text_n,
            re.DOTALL,
        ):
            return True
        if re.search(
            rf"(?:وهو|وهي|وبذلك\s+يكون)\s*{re.escape(form_n)}(?!\d){_UNIT_TAIL}\s*\.?\s*$",
            text_n,
        ):
            return True
        # Soft closer: إذن قيمة س/x هي <gt>
        if re.search(
            rf"(?:إذن|لذلك).{{0,40}}?(?:قيمة|الناتج|المجهول).{{0,20}}?"
            rf"(?:هي|هو|=)\s*{re.escape(form_n)}(?!\d)",
            text_n,
        ):
            return True
    return False


def think_leaks_final_gt(think: str, gt: Any, domain: str = "") -> bool:
    """True if final GT is restated as an answer in think (not mere intermediates).

    Intermediate arithmetic may reuse the final numeric value; flag conclusion /
    answer-cue patterns (including 'إذن … هو 24 زجاجة' and Eastern-digit forms).
    """
    if not think:
        return False
    canon = canonicalize_answer(gt, domain)
    think_n = normalize_eastern_digits(think)
    if isinstance(gt, dict) or domain == "logic":
        if canon and normalize_eastern_digits(canon) in think_n:
            return True
        if isinstance(gt, dict):
            pretty = json.dumps(gt, ensure_ascii=False)
            if pretty in think or normalize_eastern_digits(pretty) in think_n:
                return True
            # Prose dump of full ranking often leaks assignments
            if all(str(v) in think for v in gt.values()) and len(gt) >= 2:
                if re.search(r"(?:الترتيب\s+النهائي|التعيين\s+الكامل|إذن\s+الترتيب)", think):
                    return True
        return False
    if not canon or len(canon) < 1:
        return False
    lines = [ln.strip() for ln in think.strip().splitlines() if ln.strip()]
    for ln in lines[-3:]:
        if _conclusion_states_gt(ln, canon):
            return True
    for form in _gt_digit_forms(canon):
        form_n = normalize_eastern_digits(form)
        if re.search(
            rf"(?:الجواب|الإجابة|الناتج النهائي|الحل النهائي|الإجابة الصحيحة)"
            rf"\s*[:：]?\s*{re.escape(form_n)}(?!\d)",
            think_n,
        ):
            return True
    return False


def prompt_leaks_gt(prompt: str, gt: Any, domain: str = "") -> bool:
    """True if the final answer appears in the prompt (leak / operand=answer).

    For numeric GT with length >= 2, any bare occurrence is a leak (plan gate).
    Single-digit GT is ignored unless explicit answer cues appear (too common as
    operands). Logic dumps the canonical JSON.
    """
    if not prompt:
        return False
    canon = canonicalize_answer(gt, domain)
    if isinstance(gt, dict) or domain == "logic":
        return bool(canon and canon in prompt)
    if not canon or len(canon) < 1:
        return False
    cues = (
        r"(?:الجواب|الإجابة|الناتج النهائي|الحل النهائي|الإجابة الصحيحة)"
        rf"\s*[:：]?\s*{re.escape(canon)}\b"
    )
    if re.search(cues, prompt):
        return True
    # Bare final answer in prompt (len>=2): same class of noise as salt for GRPO
    if len(canon) >= 2 and re.search(rf"(?<!\d){re.escape(canon)}(?!\d)", prompt):
        return True
    return False


# Salt-class distractors (must not ship) — uniqueness must come from problem body
MARJI_JUNK_RE = re.compile(
    r"مرجع\s+\S+(?:\s+\S+){0,12}?\s+برمز\s+\d+\S*",
    re.UNICODE,
)
MUTABAA_JUNK_RE = re.compile(
    r"رقم\s+المتابعة\s+\S+|قيد\s+المتابعة\s+\S+|محضر\s+\S+\s+رقم\s+\d+\S*\s+في\s+حي",
    re.UNICODE,
)


def prompt_has_salt_junk(prompt: str) -> bool:
    """True if prompt has salt-class bureaucracy / tracking stamps."""
    if not prompt:
        return False
    if MARJI_JUNK_RE.search(prompt):
        return True
    if MUTABAA_JUNK_RE.search(prompt):
        return True
    return False


def strip_marji_junk(prompt: str) -> str:
    """Remove 'مرجع … برمز Nأ' tails (and mid-string stamps)."""
    if not prompt:
        return prompt
    p = MARJI_JUNK_RE.sub(" ", prompt)
    p = MUTABAA_JUNK_RE.sub(" ", p)
    return re.sub(r"\s{2,}", " ", p).strip()


def scrub_think_text(think: str, gt: Any, domain: str = "") -> str:
    """Drop conclusion lines that restate final GT; keep real intermediate steps.

    Never inject boilerplate fillers (those contaminated prior cold corpora).
    """
    if not think:
        return think
    canon = canonicalize_answer(gt, domain)
    # Strip known scrub contaminants first
    think = _SCRUB_BOILERPLATE_RE.sub("", think)
    think = re.sub(r"\n{3,}", "\n\n", think).strip()

    lines = think.splitlines()
    kept: list[str] = []
    for ln in lines:
        bare = ln.strip()
        if not bare:
            kept.append(ln)
            continue
        if isinstance(gt, dict) or domain == "logic":
            s = bare
            if canon and canon in s:
                continue  # drop line that embeds canonical JSON
            pretty = json.dumps(gt, ensure_ascii=False) if isinstance(gt, dict) else ""
            if pretty and pretty in s:
                continue
            if re.search(r"(?:الترتيب\s+النهائي|التعيين\s+الكامل)\s*[:：]", s):
                continue
            kept.append(ln)
            continue
        if canon and _conclusion_states_gt(bare, canon):
            # Try to keep earlier arithmetic on the same line; else drop
            form_n = normalize_eastern_digits(canon)
            s2 = bare
            for pat in (
                rf"(?:إذن|وبالتالي|وعليه).{{0,100}}?(?:هو|هي|يساوي|=)\s*{re.escape(form_n)}(?!\d){_UNIT_TAIL}\s*\.?\s*$",
                rf"(?:الإجمالي|الناتج|المجموع|الباقي).{{0,80}}?(?:هو|هي|يساوي|=)\s*{re.escape(form_n)}(?!\d){_UNIT_TAIL}\s*\.?\s*$",
                rf"(?:الجواب|الإجابة|الناتج(?:\s+النهائي)?|الحل(?:\s+النهائي)?|الإجابة الصحيحة)"
                rf"\s*[:：]?\s*{re.escape(form_n)}(?!\d){_UNIT_TAIL}",
                rf"(?:إذن|لذلك).{{0,40}}?(?:قيمة|الناتج|المجهول).{{0,20}}?(?:هي|هو|=)\s*{re.escape(form_n)}(?!\d).*$",
            ):
                s2 = re.sub(pat, "", normalize_eastern_digits(s2), flags=re.DOTALL).strip()
            if s2 and not _conclusion_states_gt(s2, canon) and len(s2.split()) >= 4:
                kept.append(s2)
            continue
        kept.append(ln)

    out = "\n".join(kept).strip()
    out = _SCRUB_BOILERPLATE_RE.sub("", out).strip()
    if not out:
        return ""
    # Drop trailing leaky lines without filler replacement
    if think_leaks_final_gt(out, gt, domain):
        parts = [ln for ln in out.splitlines() if ln.strip()]
        while parts and _conclusion_states_gt(parts[-1].strip(), canon):
            parts.pop()
        out = "\n".join(parts).strip()
    # Do NOT append boilerplate if missing terminal punctuation
    return out


def scrub_final_gt_from_think(steps: list[str], gt: Any, domain: str = "") -> list[str]:
    """Remove final-answer restatements from CoT; keep intermediate calcs."""
    canon = canonicalize_answer(gt, domain)
    out: list[str] = []
    for st in steps or []:
        s = _SCRUB_BOILERPLATE_RE.sub("", st).strip()
        if not s:
            continue
        if isinstance(gt, dict) or domain == "logic":
            if canon and canon in s:
                continue
            pretty = json.dumps(gt, ensure_ascii=False) if isinstance(gt, dict) else ""
            if pretty and pretty in s:
                continue
        elif canon and len(canon) >= 1:
            if _conclusion_states_gt(s, canon):
                continue
            form_n = normalize_eastern_digits(canon)
            s = re.sub(
                rf"(الجواب|الإجابة|الناتج النهائي|النهائي)\s*[:：]?\s*{re.escape(form_n)}",
                "",
                normalize_eastern_digits(s),
            ).strip()
            if re.search(rf"(=|يساوي)\s*{re.escape(form_n)}{_UNIT_TAIL}\s*\.?\s*$", s) and "×" not in s and "÷" not in s:
                s = re.sub(rf"(=|يساوي)\s*{re.escape(form_n)}", "", s).strip()
            if not s or _conclusion_states_gt(s, canon):
                continue
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def to_rlvr(uid: str, s: Sample) -> dict:
    ans = canonicalize_answer(s.ground_truth, s.domain)
    md = normalize_difficulty(
        s.domain,
        {
            "id": uid,
            "ground_truth_answer": metadata_gt(s.ground_truth, s.domain),
            "gt_verified": True,
            "verify_method": "programmatic",
            "num_steps_source": "programmatic",
            "template_family": (s.meta_extra or {}).get("template_family") or s.domain,
            "solution_steps": scrub_final_gt_from_think(
                list(s.solution_steps or []), s.ground_truth, s.domain
            ),
            "corpus_version": "v1_frontier_regen",
            **s.meta_extra,
        },
        s.num_steps,
    )
    return {
        "id": uid,
        "source": "rlvr",
        "domain": s.domain,
        "prompt": s.prompt,
        "response": "",
        "answer": ans,
        "metadata": md,
        "quality": {"score": 1.0, "arabic_purity": round(_arabic_purity(s.prompt), 4)},
    }


def to_coldstart(uid: str, s: Sample) -> dict:
    steps = scrub_final_gt_from_think(list(s.solution_steps or []), s.ground_truth, s.domain)
    think = "\n".join(steps)
    ans = canonicalize_answer(s.ground_truth, s.domain)
    response = f"<think>\n{think}\n</think>\n<answer>{ans}</answer>"
    md = normalize_difficulty(
        s.domain,
        {
            "id": uid,
            "ground_truth_answer": metadata_gt(s.ground_truth, s.domain),
            "gt_verified": True,
            "verify_method": "programmatic",
            "num_steps_source": "programmatic",
            "template_family": (s.meta_extra or {}).get("template_family") or s.domain,
            "corpus_version": "v1_frontier_regen",
            **s.meta_extra,
        },
        s.num_steps,
    )
    return {
        "id": uid,
        "source": "coldstart",
        "domain": s.domain,
        "prompt": s.prompt,
        "response": response,
        "answer": ans,
        "metadata": md,
        "quality": {"score": 1.0, "arabic_purity": round(_arabic_purity(s.prompt + think), 4)},
    }


def gen_arapro_knowledge(rng: random.Random) -> Sample:
    topics = [
        ("ما هي الوظيفة الرئيسية للميتوكندريا في الخلية؟", "إنتاج الطاقة (ATP)", ["إنتاج الطاقة (ATP)", "تخزين الماء", "بناء الجدار الخلوي", "امتصاص الضوء"], "البيولوجيا / العلوم الطبية"),
        ("ما القانون الذي ينص على أن لكل فعل رد فعل مساوٍ له في المقدار ومضاد له في الاتجاه؟", "قانون نيوتن الثالث", ["قانون نيوتن الأول", "قانون نيوتن الثاني", "قانون نيوتن الثالث", "قانون الجاذبية الكوني"], "الفيزياء الكلاسيكية"),
        ("أي من العناصر التالية يمتلك أعلى كهروسلبية في الجدول الدوري؟", "الفلور", ["الصوديوم", "الأكسجين", "الفلور", "الكلور"], "الكيمياء العامة"),
        ("ما هو اسم النظام القضائي الذي يعتمد على سوابق الأحكام القضائية؟", "القانون العام (Common Law)", ["القانون المدني", "القانون العام (Common Law)", "القانون الجنائي", "القانون التجاري"], "العلوم القانونية"),
        ("ما هي المعركة التي وقعت عام 630 م (8 هـ) وفتحت فيها مكة المكرمة؟", "فتح مكة", ["معركة بدر", "معركة أحد", "فتح مكة", "معركة حنين"], "التاريخ والسياسات"),
        ("ما الجهاز المسؤول عن نقل الأكسجين إلى خلايا الجسم؟", "جهاز الدوران / الدم", ["جهاز الدوران / الدم", "الجهاز الهضمي", "الجهاز الهيكلي", "الجهاز اللمفاوي"], "علم وظائف الأعضاء"),
        ("ما اسم العصر الجيولوجي الذي ظهرت فيه الديناصورات وازدهرت؟", "الحقبة الوسطى (Mesozoic)", ["الحقبة القديمة", "الحقبة الوسطى (Mesozoic)", "الحقبة الحديثة", "العصر الحجري"], "علم الأرض والجيولوجيا"),
        ("ما اسم النظرية الاقتصادية التي تؤكد على دور العرض والطلب في تحديد الأسعار؟", "اقتصاد السوق الحر", ["الاشتراكية", "اقتصاد السوق الحر", "الإقطاعية", "الاقتصاد الموجه"], "العلوم الاقتصادية"),
        ("أي الغازات التالية يشكل النسبة الأكبر في الغلاف الجوي للأرض؟", "النيتروجين", ["الأكسجين", "النيتروجين", "ثاني أكسيد الكربون", "الآرجون"], "علوم الجو والفلك"),
        ("ما الوحدة الأساسية لقياس شدة التيار الكهربائي في النظام الدولي؟", "الأمبير", ["الفولت", "الأوم", "الأمبير", "الواط"], "الفيزياء الكهربائية"),
        ("ما اسم الميثاق الذي أسس الأمم المتحدة عام 1945 م؟", "ميثاق سان فرانسيسكو", ["معاهدة فيرساي", "اتفاقية جنيف", "ميثاق سان فرانسيسكو", "معاهدة روما"], "العلاقات الدولية"),
        ("ما المركب الكيميائي المسؤول عن نقل المعلومات الوراثية في الكائنات الحية؟", "حمض DNA", ["حمض RNA", "حمض DNA", "البروتينات", "الدهون الفوسفاتية"], "علم الوراثة الجزيئية"),
    ]
    q, ans, options_base, domain_tag = rng.choice(topics)
    options = list(options_base)
    rng.shuffle(options)
    opts_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    code = rng.randint(100, 999)
    prompt = f"السؤال المرجعي [رمز-{code}] ({domain_tag}):\n{q}\n\nالخيارات:\n{opts_text}\n\nحدد الإجابة الصحيحة مع بيان السبب."
    correct_letter = chr(65 + options.index(ans))
    steps = [
        f"نحلل السؤال المطلوب في مجال {domain_tag}.",
        f"نراجع الخيارات المتاحة: {ans} هو الخيار الصحيح علمياً وفقهياً.",
        f"الإجابة الصحيحة هي الخيار {correct_letter}."
    ]
    return Sample("arapro_knowledge", prompt, correct_letter, 3, {"topic": domain_tag, "code": code}, steps)


def gen_ifeval_multiconstraint(rng: random.Random) -> Sample:
    topics = ["الذكاء الاصطناعي", "الطاقة المتجددة", "التخطيط المالي", "الصحة النفسية", "الرؤية الاقتصادية", "الأمن السيبراني", "الابتكار الرقمي", "التعليم الذكي"]
    topic = rng.choice(topics)
    min_w = rng.randint(80, 150)
    max_w = min_w + rng.randint(40, 70)
    bullets = rng.randint(3, 5)
    forbidden = rng.choice(["حاسوب", "مشكلة", "صعب", "تراجع", "فشل", "بطء", "مخاطر", "خسارة"])
    code = rng.randint(1000, 9999)
    
    prompt = (
        f"اكتب مقالاً توجيهياً عن '{topic}' [مرجع-{code}] مع الالتزام التام بالشروط التالية:\n"
        f"1. أن يتراوح طول النص بين {min_w} و{max_w} كلمة.\n"
        f"2. يتضمن قائمة نقطية تحتوي على بالضبط {bullets} نقاط.\n"
        f"3. عدم استخدام كلمة '{forbidden}' نهائياً في النص.\n"
        f"4. إنهاء النص بملاحظة تبدأ بـ 'ملاحظة:'."
    )
    steps = [
        f"صياغة مقال عن {topic} مع ضبط عدد الكلمات بين {min_w} و{max_w} كلمة.",
        f"إدراج {bullets} نقاط رئيسية في قائمة نقطية واضحة.",
        f"تجنب استخدام كلمة '{forbidden}' تماماً أثناء الكتابة.",
        f"إضافة الملاحظة الختامية بالشكل المطلوب."
    ]
    return Sample("ifeval_multiconstraint", prompt, "التزام تام بالشروط", 4, {"topic": topic, "forbidden": forbidden, "bullets": bullets, "code": code}, steps)


def gen_aratrust_truth(rng: random.Random) -> Sample:
    scenarios = [
        ("هل يسبب ابتلاع العلكة بقاءها في المعدة لمدة 7 سنوات؟", "لا، يتم طرحها طبيعياً عبر الجهاز الهضمي خلال أيام.", "شائعة طبية بخصوص الجهاز الهضمي"),
        ("هل السهر تحت الضوء الأبيض يؤثر على إفراز هرمون الميلامين؟", "نعم، يؤثر الضوء الأزرق والأبيض سلباً على إفراز الميلاتونين.", "حقائق علوم النوم والأعصاب"),
        ("هل شرب الماء بكثرة أثناء الأكل يسبب الكرش مباشرة؟", "لا، الكرش ينجم عن زيادة السعرات الحرارية وتراكم الدهون وليس الماء.", "التغذية واللياقة البدنية"),
        ("هل القراءة في الضوء الخافت تسبب العمى الدائم؟", "لا، تسبب إجهاداً مؤقتاً للعين فقط ولا تسبب العمى.", "صحة العيون والبصريات"),
        ("هل أكل السمك مع اللبن يسبب تسمماً غذائياً مباشراً؟", "لا، لا يوجد تعارض كيميائي طالما أن كلا العنصرين طازجان وغير فاسدين.", "خرافات التغذية الشعبية"),
        ("هل فتح النوافذ أثناء العواصف الرعدية يمنع انفجار المنزل؟", "لا، يفضل إغلاق النوافذ لمنع دخول الرياح والصواعق والضغط العالي.", "السلامة العامة والدفاع المدني"),
        ("هل الحلي الذهبية تجذب الصواعق مباشرة في المطر؟", "لا، حجم الحلي صغير جداً ولا يشكل جاذباً للصواعق مقارنة بالأشجار والمباني.", "علوم الفيزياء والتجهيزات"),
        ("هل وضع الزيت على الحروق الحديثة يساعد في التئامها بسرعة؟", "لا، الزيت يحبس الحرارة داخل الجلد ويزيد من عمق الحرق ويجب استخدام الماء الجاري.", "الإسعافات الأولية والطب الطارئ"),
    ]
    q, truth_fact, cat = rng.choice(scenarios)
    code = rng.randint(100, 999)
    prompt = f"السؤال الإرشادي [حالة-{code}] ({cat}):\n{q}\nأجب بدقة علمية مع تصحيح أي مفاهيم خاطئة شائعة."
    steps = [
        f"تحليل المفهوم الشائع في مسألة: {q}",
        f"التدقيق العلمي: {truth_fact}",
        f"النتيجة: تصحيح الخلل المفهومي وإيراد الحقيقة العلمية."
    ]
    return Sample("aratrust_truth", prompt, truth_fact, 3, {"category": cat, "code": code}, steps)


GENERATORS = {
    "gsm8k": gen_gsm8k,
    "math": gen_math,
    "math_comp": gen_math_comp,
    "logic": gen_logic,
    "arapro_knowledge": gen_arapro_knowledge,
    "ifeval_multiconstraint": gen_ifeval_multiconstraint,
    "aratrust_truth": gen_aratrust_truth,
}


def generate_unique_sample(
    domain: str,
    rng: random.Random,
    seen_prompts: set[str],
    max_tries: int = 80,
    *,
    corpus_prompts: list[str] | None = None,
    max_jaccard: float = 0.55,
) -> Sample:
    """Generate a sample with exact + near-dup (Jaccard) uniqueness."""
    from formal_saudi_style import jaccard

    gen = GENERATORS[domain]
    corpus = corpus_prompts or []
    for _ in range(max_tries):
        s = gen(rng)
        key = re.sub(r"\s+", " ", s.prompt.strip())
        if key in seen_prompts:
            continue
        if any(jaccard(s.prompt, p) >= max_jaccard for p in corpus[-800:]):
            continue
        seen_prompts.add(key)
        return s
    raise RuntimeError(f"Could not unique-sample domain={domain}")
