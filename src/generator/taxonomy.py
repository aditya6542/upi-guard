"""Fraud and legitimate transaction taxonomy.

This is the contract for the whole project: the generator emits transactions from
these definitions, and every detector rule maps back to a type here. Adding a
scam type means adding an entry here first — nothing else needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScamType:
    """One fraud pattern.

    `message_templates` carry the social-engineering script. `marker_tokens` are
    the words a deterministic rule can key on without a trained model — the
    rule engine uses these, the similarity engine uses the whole template.
    """

    code: str
    label: str
    # What the victim believes is happening vs what actually happens. Shown in the
    # warning screen — this gap is the actual attack.
    claimed_intent: str
    actual_effect: str
    direction: str  # COLLECT_REQUEST | PAY
    message_templates: tuple[str, ...]
    marker_tokens: tuple[str, ...]
    amount_range: tuple[int, int]
    round_amount_bias: float = 0.6
    # Fraud accounts are usually new. This drives the account-age feature.
    account_age_days: tuple[int, int] = (0, 30)
    urgency: float = 0.7


@dataclass(frozen=True)
class LegitType:
    """One benign pattern. Needed so false-positive rate is measured, not assumed.

    `account_age_days` deliberately overlaps the fraud range for most types.
    Real users open accounts constantly, and a detector that only learns
    "new account == fraud" would score perfectly on synthetic data and fail on
    production traffic.
    """

    code: str
    label: str
    direction: str  # PAY | COLLECT_REQUEST | INCOMING
    message_templates: tuple[str, ...]
    amount_range: tuple[int, int]
    round_amount_bias: float = 0.3
    account_age_days: tuple[int, int] = (5, 3650)


# --------------------------------------------------------------------- FRAUD
SCAM_TYPES: tuple[ScamType, ...] = (
    ScamType(
        code="F01",
        label="Collect request disguised as refund",
        claimed_intent="A merchant is returning excess money to you",
        actual_effect="You send the money instead of receiving it",
        direction="COLLECT_REQUEST",
        message_templates=(
            "Dear customer, your refund of Rs {amount} has been credited. Kindly accept the request to receive it.",
            "Your payment of Rs {amount} was debited by mistake. Accept the collect request to get Rs {amount} refunded.",
            "Rs {amount} refund processed in your favour. Please accept this request within 24 hours to credit your account.",
            "Amount of Rs {amount} has been reversed. Accept the collect request for instant refund.",
        ),
        marker_tokens=("refund", "credit", "reversed", "accept", "collect request", "in your favour"),
        amount_range=(500, 12000),
        round_amount_bias=0.75,
        account_age_days=(0, 21),
        urgency=0.85,
    ),
    ScamType(
        code="F02",
        label="QR code swap / merchant impersonation",
        claimed_intent="Paying a shop for a purchase",
        actual_effect="Payment goes to the fraudster's personal QR instead",
        direction="PAY",
        message_templates=(
            "Scan this QR to complete your payment of Rs {amount} for the order.",
            "Pay Rs {amount} using this QR code and share the screenshot once done.",
            "Merchant QR updated. Please scan and pay Rs {amount} for your booking.",
            "Kindly scan the QR and make the payment of Rs {amount} to confirm your order.",
        ),
        marker_tokens=("scan", "qr code", "qr", "screenshot", "merchant", "confirm your order"),
        amount_range=(200, 25000),
        round_amount_bias=0.5,
        account_age_days=(0, 45),
        urgency=0.7,
    ),
    ScamType(
        code="F03",
        label="Screen-sharing / remote-access scam",
        claimed_intent="Bank is helping you resolve a complaint",
        actual_effect="You hand over control and the fraudster moves your money",
        direction="COLLECT_REQUEST",
        message_templates=(
            "Sir, for your KYC update please install this app and share your screen. Your account of Rs {amount} will be credited.",
            "We need remote access to verify your account. Accept the request of Rs {amount} after sharing the screen.",
            "To resolve the pending complaint, share your screen with us. The refund of Rs {amount} will follow.",
            "Kindly enable screen sharing so our executive can process the Rs {amount} credit to your account.",
        ),
        marker_tokens=("share your screen", "screen sharing", "remote access", "install", "executive", "kyc"),
        amount_range=(1000, 50000),
        round_amount_bias=0.6,
        account_age_days=(1, 60),
        urgency=0.9,
    ),
    ScamType(
        code="F04",
        label="Fake KYC / bank account update",
        claimed_intent="Routine compliance check by your bank",
        actual_effect="You submit credentials or approve a collect request",
        direction="COLLECT_REQUEST",
        message_templates=(
            "Your account will be suspended unless KYC is updated. Share OTP to complete the Rs {amount} verification.",
            "Immediate KYC pending. Update your details and accept the request of Rs {amount} to stay active.",
            "As per RBI guidelines your Aadhaar must be re-verified. Submit details to claim Rs {amount}.",
            "Your bank account is unverified. Confirm KYC now to avoid a penalty of Rs {amount}.",
        ),
        marker_tokens=("kyc", "kyc update", "suspended", "otp", "rbi", "aadhaar", "verification"),
        amount_range=(100, 5000),
        round_amount_bias=0.5,
        account_age_days=(0, 90),
        urgency=0.95,
    ),
    ScamType(
        code="F05",
        label="Mule chain / pass-through laundering",
        claimed_intent="Passing money through your account for someone",
        actual_effect="Your account is used to receive then forward scam money",
        direction="INCOMING",
        message_templates=(
            "Received Rs {amount} in your account. Kindly transfer the same amount to the account shared below.",
            "Rs {amount} credited to your UPI. Please forward it to the given ID within 30 minutes.",
            "This is a refund for your cancelled order. Send Rs {amount} to the shared number immediately.",
            "Amount of Rs {amount} has been deposited. Transfer it onwards as instructed to confirm your order.",
        ),
        marker_tokens=("transfer", "forward", "shared below", "within 30 minutes", "onwards", "credited to your upi"),
        amount_range=(1000, 80000),
        round_amount_bias=0.7,
        account_age_days=(1, 120),
        urgency=0.95,
    ),
    ScamType(
        code="F06",
        label="OTP / UPI PIN harvesting",
        claimed_intent="Confirming that you are the account holder",
        actual_effect="You read out your OTP or PIN and the payment goes through",
        direction="COLLECT_REQUEST",
        message_templates=(
            "Share the OTP received on your phone to verify your account of Rs {amount}.",
            "Enter the 6-digit UPI PIN in the link below to authorise the refund of Rs {amount}.",
            "To confirm your identity, tell us the OTP we just sent. Your Rs {amount} credit is on hold.",
            "Share your UPI PIN to authorise the Rs {amount} transaction. It will be refunded immediately.",
        ),
        marker_tokens=("otp", "upi pin", "6-digit", "verify", "share the otp", "authorise"),
        amount_range=(100, 10000),
        round_amount_bias=0.4,
        account_age_days=(0, 60),
        urgency=0.95,
    ),
    ScamType(
        code="F07",
        label="Overpayment refund scam",
        claimed_intent="Returning money you were overcharged by",
        actual_effect="A fake overpayment forces you to send real money back",
        direction="COLLECT_REQUEST",
        message_templates=(
            "You have been overcharged. Refund the excess amount of Rs {amount} to complete the cancellation.",
            "Rs {amount} was charged extra. Please return the difference to the original payment source.",
            "An extra Rs {amount} was debited by mistake. Kindly refund it so the payment can be reversed.",
            "Refund the excess Rs {amount} within 10 minutes or the order will be cancelled and forfeited.",
        ),
        marker_tokens=("overcharged", "excess", "refund the difference", "extra", "reverse"),
        amount_range=(300, 9000),
        round_amount_bias=0.65,
        account_age_days=(0, 40),
        urgency=0.8,
    ),
    ScamType(
        code="F08",
        label="Lottery / prize / tournament bait",
        claimed_intent="Claiming a prize you supposedly won",
        actual_effect="You pay a processing fee to unlock a prize that does not exist",
        direction="COLLECT_REQUEST",
        message_templates=(
            "Congratulations! You have won Rs {amount} in the monthly lottery. Pay the processing fee to claim it.",
            "You are selected for the prize of Rs {amount}. Kindly accept the request of Rs {amount} to release the amount.",
            "Your number has won Rs {amount}. Complete the payment of Rs {amount} to register the claim.",
            "Winner alert! Rs {amount} prize money awaits. Pay the registration charge to receive it.",
        ),
        marker_tokens=("congratulations", "won", "lottery", "prize", "claim", "processing fee", "winner"),
        amount_range=(500, 30000),
        round_amount_bias=0.8,
        account_age_days=(0, 25),
        urgency=0.85,
    ),
    ScamType(
        code="F09",
        label="Guaranteed-return investment",
        claimed_intent="Investing in a high-return scheme",
        actual_effect="You deposit money that is never returned",
        direction="PAY",
        message_templates=(
            "Guaranteed 30% monthly return. Invest Rs {amount} today and start earning from tomorrow.",
            "Our AI-driven scheme doubles your money in 15 days. Pay Rs {amount} to start your portfolio.",
            "Invest Rs {amount} for a risk-free 25% return. Withdrawal allowed after 7 days.",
            "Join 5000 investors earning daily. Deposit Rs {amount} to activate your account.",
        ),
        marker_tokens=("guaranteed", "return", "invest", "risk-free", "withdraw", "roi", "portfolio"),
        amount_range=(2000, 200000),
        round_amount_bias=0.7,
        account_age_days=(0, 180),
        urgency=0.6,
    ),
    ScamType(
        code="F10",
        label="Fake job / task-based earning",
        claimed_intent="Earning money by completing simple tasks",
        actual_effect="You pay a registration deposit that never returns",
        direction="PAY",
        message_templates=(
            "Part-time work from home. Complete 3 tasks daily and earn Rs {amount}. Pay Rs {amount} to register.",
            "You have been selected for a work-from-home role. Deposit Rs {amount} to activate your account.",
            "Simple data entry task pays Rs {amount} per task. Register now with Rs {amount} to start.",
            "Liked your profile for the task role. Pay Rs {amount} joining fee to receive your first payout.",
        ),
        marker_tokens=("work from home", "part-time", "task", "register", "joining fee", "deposit", "earn"),
        amount_range=(500, 25000),
        round_amount_bias=0.75,
        account_age_days=(0, 90),
        urgency=0.65,
    ),
)


# ------------------------------------------------------------------ LEGITIMATE
# Account ages overlap the fraud range on purpose (see LegitType docstring).
# L04/L05/L07 stay old: school-fee and salary traffic genuinely comes from
# long-lived accounts, and keeping a few of them anchors the age feature.
LEGIT_TYPES: tuple[LegitType, ...] = (
    LegitType("L01", "Electricity bill", "PAY", (
        "Your electricity bill of Rs {amount} is due. Pay via UPI.",
        "BESCOM bill of Rs {amount} pending. Please pay before the due date.",
        "Monthly power bill Rs {amount}. Tap to pay.",
    ), (400, 8000), 0.4, (12, 3650)),
    LegitType("L02", "Mobile recharge", "PAY", (
        "Recharge of Rs {amount} for your number is ready.",
        "Pay Rs {amount} to top up your mobile balance.",
        "DTH recharge of Rs {amount} due.",
    ), (99, 2000), 0.8, (20, 3650)),
    LegitType("L03", "Friend dinner split", "COLLECT_REQUEST", (
        "Dinner at Cafe Blue — my share Rs {amount}. UPI bhej do.",
        "Splitting the cab fare, your part Rs {amount}.",
        "Groceries split — yours is Rs {amount}.",
    ), (100, 6000), 0.3, (30, 3650)),
    LegitType("L04", "School fees", "PAY", (
        "School fees of Rs {amount} for this term is due.",
        "Fee payment request of Rs {amount} from the school portal.",
        "Term fees Rs {amount} — pay before the 15th.",
    ), (5000, 80000), 0.5, (200, 3650)),
    LegitType("L05", "EMI instalment", "PAY", (
        "EMI auto-debit of Rs {amount} will be charged today.",
        "Loan EMI of Rs {amount} is due. Keep your balance sufficient.",
        "Monthly instalment Rs {amount} scheduled.",
    ), (1500, 60000), 0.7, (90, 3650)),
    LegitType("L06", "Rent transfer", "PAY", (
        "Rent for this month Rs {amount} — please pay before the 5th.",
        "House rent transfer Rs {amount}.",
        "Maalik ka rent Rs {amount} due hai.",
    ), (5000, 50000), 0.6, (60, 3650)),
    LegitType("L07", "Salary credit notice", "INCOMING", (
        "Salary of Rs {amount} has been credited to your account.",
        "Your monthly pay of Rs {amount} has been deposited.",
    ), (15000, 300000), 0.4, (400, 3650)),
    # The deliberate false-positive trap: a genuine payment that uses urgency
    # language and a new device. A detector that keys on urgency alone will
    # block this, and that error is more costly than the fraud it prevents.
    LegitType("L08", "Medical emergency", "PAY", (
        "Hospital advance of Rs {amount} — urgent, please pay now.",
        "Mummy ke liye medicine Rs {amount} bhejo urgently.",
        "Diagnostic charges Rs {amount} paid at the hospital, sharing with you.",
    ), (500, 60000), 0.3, (3, 3650)),
    LegitType("L09", "Insurance premium", "PAY", (
        "Insurance premium of Rs {amount} is due for renewal.",
        "Policy premium Rs {amount} — pay to avoid lapse.",
        "Life insurance due date is near. Pay Rs {amount}.",
    ), (2000, 30000), 0.7, (150, 3650)),
    LegitType("L10", "Small merchant purchase", "PAY", (
        "Payment of Rs {amount} to Metro Mart.",
        "Paid Rs {amount} at the grocery store.",
        "Tea and snacks Rs {amount}.",
    ), (20, 5000), 0.2, (10, 3650)),
)


SCAM_BY_CODE = {s.code: s for s in SCAM_TYPES}
LEGIT_BY_CODE = {l.code: l for l in LEGIT_TYPES}
