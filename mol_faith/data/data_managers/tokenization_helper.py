import re
from rdkit import Chem

from typing import Dict, List, Optional


def test_round_trip_fidelity(tokenizer, smiles_list: List[str],
                             max_length: int = 512,
                             sample_size: Optional[int] = None) -> Dict:
    """
    Test encode-decode round-trip fidelity to ensure no information loss.

    Args:
        tokenizer: Built tokenizer instance
        smiles_list: List of SMILES strings to test
        max_length: Max length for encoding
        sample_size: If specified, test only a random sample
    Returns:
        Dictionary with fidelity test results
    """
    if not tokenizer.token_to_id:
        raise ValueError("Tokenizer vocabulary not built. Call build_vocab_from_corpus first.")

    # Sample if requested
    if sample_size and len(smiles_list) > sample_size:
        import random
        test_smiles = random.sample(smiles_list, sample_size)
        print(f"Testing random sample of {sample_size} SMILES")
    else:
        test_smiles = smiles_list
        print(f"Testing all {len(test_smiles)} SMILES")

    results = {
        'total_tested': len(test_smiles),
        'perfect_matches': 0,
        'failed_roundtrips': [],
        'truncation_losses': [],
        'tokenization_failures': [],
        'unknown_token_issues': []
    }

    print("Testing encode-decode round-trip fidelity...")

    for i, original_smiles in enumerate(test_smiles):
        try:
            # Step 1: Tokenize
            tokens = tokenizer.tokenize(original_smiles)

            # Step 2: Encode to IDs
            token_ids = tokenizer.encode(original_smiles, max_seq_len=max_length)

            # Step 3: Decode back to SMILES
            decoded_smiles = tokenizer.decode(token_ids, skip_special_tokens=True)

            # Step 4: Compare
            if original_smiles == decoded_smiles:
                results['perfect_matches'] += 1
            else:
                # Analyze the type of failure
                failure_info = {
                    'index': i,
                    'original': original_smiles,
                    'decoded': decoded_smiles,
                    'original_length': len(tokens),
                    'tokens': tokens,
                    'token_ids': token_ids.tolist()[:20]  # First 20 for debugging
                }

                # Check if truncation caused the issue
                if len(tokens) > max_length:
                    results['truncation_losses'].append(failure_info)
                # Check for unknown tokens
                elif tokenizer.unk_token_id in token_ids:
                    results['unknown_token_issues'].append(failure_info)
                else:
                    results['failed_roundtrips'].append(failure_info)

        except Exception as e:
            results['tokenization_failures'].append({
                'index': i,
                'smiles': original_smiles,
                'error': str(e)
            })

    # Calculate statistics
    total_successful = results['perfect_matches']
    success_rate = total_successful / results['total_tested']
    perfect_rate = results['perfect_matches'] / results['total_tested']

    results['success_rate'] = success_rate
    results['perfect_success_rate'] = perfect_rate
    results['failure_breakdown'] = {
        'truncation_losses': len(results['truncation_losses']),
        'unknown_token_issues': len(results['unknown_token_issues']),
        'other_failures': len(results['failed_roundtrips']),
        'tokenization_failures': len(results['tokenization_failures']),
    }

    # Print summary
    _print_fidelity_summary(results)

    return results


def _print_fidelity_summary(results: Dict):
    """Print a summary of the fidelity test results."""
    print(f"\nROUND-TRIP FIDELITY TEST RESULTS")
    print("=" * 50)
    print(f"Total SMILES tested: {results['total_tested']}")
    print(f"Perfect matches: {results['perfect_matches']} ({results['perfect_success_rate'] * 100:.2f}%)")

    if results['success_rate'] == 1.0:
        print("🎉 PERFECT FIDELITY! All SMILES round-tripped correctly.")
    else:
        remaining_failures = (results['total_tested'] -
                              results['perfect_matches'])
        print(f"\n❌ Found {remaining_failures} real mismatches:")

        breakdown = results['failure_breakdown']
        if breakdown['truncation_losses'] > 0:
            print(f"  - {breakdown['truncation_losses']} due to sequence truncation")
        if breakdown['unknown_token_issues'] > 0:
            print(f"  - {breakdown['unknown_token_issues']} due to unknown tokens")
        if breakdown['other_failures'] > 0:
            print(f"  - {breakdown['other_failures']} due to other issues")
        if breakdown['tokenization_failures'] > 0:
            print(f"  - {breakdown['tokenization_failures']} tokenization failures")

    # Show examples of real failures with detailed analysis
    if results['failed_roundtrips']:
        print(f"\n❌ REAL ROUND-TRIP FAILURES (need investigation):")
        for i, failure in enumerate(results['failed_roundtrips'][:5], 1):  # Show up to 5 examples
            print(f"\n  Example {i}:")
            print(f"    Original:  '{failure['original']}'")
            print(f"    Decoded:   '{failure['decoded']}'")
            print(f"    Tokens:    {failure['tokens']}")

            # Analyze the difference
            orig_tokens = failure['tokens']
            try:
                # Re-decode to see what tokens actually produced
                reconstructed = ''.join(orig_tokens)
                print(f"    Reconstructed from tokens: '{reconstructed}'")

                # Show character-by-character diff if short enough
                if len(failure['original']) < 50 and len(failure['decoded']) < 50:
                    orig_chars = list(failure['original'])
                    decoded_chars = list(failure['decoded'])
                    print(f"    Original chars:  {orig_chars}")
                    print(f"    Decoded chars:   {decoded_chars}")

                    # Find first difference
                    for j, (o, d) in enumerate(zip(orig_chars, decoded_chars)):
                        if o != d:
                            print(f"    First diff at pos {j}: '{o}' vs '{d}'")
                            break

                    if len(orig_chars) != len(decoded_chars):
                        print(f"    Length diff: {len(orig_chars)} vs {len(decoded_chars)}")

            except Exception as e:
                print(f"    Analysis error: {e}")

    # Show truncation examples
    if results['truncation_losses']:
        print(f"\n📏 TRUNCATION LOSSES (sequence too long):")
        for i, loss in enumerate(results['truncation_losses'][:3], 1):
            print(f"\n  Example {i}:")
            print(f"    Original ({loss['original_length']} tokens): '{loss['original'][:80]}...'")
            print(f"    Decoded: '{loss['decoded']}'")
            print(f"    Problem: Original has {loss['original_length']} tokens, exceeds max_length")

    # Show unknown token examples
    if results['unknown_token_issues']:
        print(f"\n🔤 UNKNOWN TOKEN ISSUES (vocabulary incomplete):")
        for i, issue in enumerate(results['unknown_token_issues'][:3], 1):
            print(f"\n  Example {i}:")
            print(f"    Original: '{issue['original']}'")
            print(f"    Tokens: {issue['tokens']}")

    # Show tokenization failure examples
    if results['tokenization_failures']:
        print(f"\n⚠️  TOKENIZATION FAILURES (SMILES parsing errors):")
        for i, failure in enumerate(results['tokenization_failures'][:3], 1):
            print(f"\n  Example {i}:")
            print(f"    SMILES: '{failure['smiles']}'")
            print(f"    Error: {failure['error']}")

            # Basic analysis of the problematic SMILES
            smiles = failure['smiles']
            print(f"    Length: {len(smiles)} characters")
            print(f"    Contains: {set(smiles)}")

            # Check for obvious issues
            if not smiles.strip():
                print(f"    Issue: Empty or whitespace-only SMILES")
            elif len(smiles) > 1000:
                print(f"    Issue: Extremely long SMILES")
            elif any(ord(c) > 127 for c in smiles):
                print(f"    Issue: Contains non-ASCII characters")


def extract_elements_from_smiles(smiles_list):
    """
    Extract all unique chemical elements from a list of SMILES strings using RDKit.
    This is the correct way - RDKit properly parses the molecules.

    Args:
        smiles_list (list): List of SMILES strings

    Returns:
        set: Set of unique chemical elements found
    """
    all_elements = set()
    failed_smiles = []

    for smiles in smiles_list:
        if not smiles:
            continue

        try:
            # Parse SMILES into molecule object
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                failed_smiles.append(smiles)
                continue

            # Get all atoms and their element symbols
            for atom in mol.GetAtoms():
                element_symbol = atom.GetSymbol()
                all_elements.add(element_symbol)

        except Exception as e:
            failed_smiles.append(f"{smiles} (error: {e})")

    result = {
        'elements': sorted(all_elements),
        'failed_smiles': failed_smiles
    }

    return result
